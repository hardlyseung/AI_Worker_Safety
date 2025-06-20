import cv2
import mediapipe as mp  # # PoseLandmark 이름을 가져오기 위해 필요
import time
import os
import sys # 시스템 정보 (파이썬 버전 등) 확인용
import tqdm
import numpy as np      # NumPy: 다차원 배열, 행렬 연산
import pandas as pd     # Pandas: 데이터 조작, 분석
import glob             # Glob: 파일 경로 패턴 매칭
import torch            # PyTorch: 딥러닝 프레임워크
import torch.nn as nn   # PyTorch: 신경망 모듈
import torch.optim as optim # PyTorch: 최적화 알고리즘
import tensorflow as tf, numpy as np, datetime, os
import matplotlib.pyplot as plt # 임계값 설정을 위한 시각화
from collections import deque
import subprocess
import RPi.GPIO as GPIO
import datetime

# 프레임 초기화 딕셔너리
person_state = {}
person_prev_joints = {}
person_joint_buffer = {}

#-------------------log_print------------------------
last_log_time = time.time()

def log_status_every_3s(frame_idx, detected, person_id=None):
    global last_log_time
    now = time.time()
    if now - last_log_time >= 3:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        if detected:
            print(f"[{timestamp}] Frame {frame_idx}: Person detected  | ID: {person_id}")
        else:
            print(f"[{timestamp}] Frame {frame_idx}: No person detected")
        last_log_time = now


WIDTH, HEIGHT = 640, 480

#------------------ YOLOv5 모델 로드  ------------------
yolo_model = torch.hub.load('/home/py/yolov5', 'custom', path= 'yolov5s.pt', source='local') # v5s모델 씀
yolo_model.classes = [0]  # 사람만 감지
yolo_model.conf = 0.5

#model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
#model.conf = 0.5
#model.classes = [0]  # 사람만 감지 

#------------------ LED 및 부저 함수 ------------------
BUZZER_PIN = 17  # 부저 연결 핀
LED_PIN = 27 # LED 연결 핀
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)

def alert_buzzer_and_led():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(0.3)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.output(LED_PIN, GPIO.LOW)


#------------------ MediaPipe 초기화 ------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
max_frames = 30     # 30프레임 (25~30으로 유동적 조정 가능)
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

# 관절 인덱스가 부족할 시 -1.0으로 처리
def extract_20joints_from_landmarks(landmarks, joint_map):
    joint_data = []
    for joint_name, info in joint_map.items():
        if isinstance(info, tuple) and info[0] == "avg":
            indices = info[1]
            if all(i < len(landmarks) for i in indices):
                joint = np.mean([landmarks[i] for i in indices], axis=0)
            else:
                joint = np.array([-1.0, -1.0, -1.0])
        else:
            if info < len(landmarks):
                joint = landmarks[info]
            else:
                joint = np.array([-1.0, -1.0, -1.0])
        joint_data.extend(joint.tolist())
    return joint_data

#------------------ 관절 매핑 (20개) ------------------
mediapipe_index = {
    "Head": 0,
    "Shouldercentre": ("avg", [11, 14]),
    "Spine": ("avg", [27, 28]),
    "Leftshoulder": 14,
    "Leftelbow": 15,
    "Lefthand": 16,
    "Rightshoulder": 11,
    "Rightelbow": 12,
    "Righthand": 13,
    "Lefthip": 27,
    "Leftknee": 29,
    "Leftfoot": 31,
    "Righthip": 28,
    "Rightknee": 30,
    "Rightfoot": 32,
    "Hipcentre": ("avg", [27, 28]),
    "Leftwrist": 16,
    "Leftankle": 31,
    "Rightwrist": 13,
    "Rightankle": 32
}

#------------------ LSTM 오토인코더 ------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, dropout=0.3, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, dropout=0.3, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (h_n, c_n) = self.encoder(x)
        decoder_input = torch.zeros(batch_size, seq_len, h_n.shape[-1]).to(x.device)
        dec_out, _ = self.decoder(decoder_input, (h_n, c_n))
        dec_out = self.layer_norm(dec_out)  # 안정화
        out = self.output_layer(dec_out)

        return out

#------------------ 하이퍼파라미터 설정 ------------------ 
input_dim = 60  # 20 joints * 3
hidden_dim = 64
seq_len = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAutoencoder(input_dim, hidden_dim).to(device)
model.load_state_dict(torch.load("/home/py/lstm_autoencoder.pth", map_location=device)) #파일경로
model.eval()

#------------------ 위험 동작 감지 ------------------ 
N_JOINTS = 20  # 관절 개수
SEQ_LEN = 30   # 30프레임
STATE_NORMAL = 0       # 서있음
STATE_LYING = 1        # 누워있는 상태
STATE_POSSIBLE_FALL = 2  # 낙상 의심 (누워서 정지)

def estimate_posture_from_bbox(bbox, standing_thresh=1.2, lying_thresh=0.8):
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # 예외 처리: bbox가 유효하지 않거나 크기가 너무 작을 경우
    if width <= 0 or height <= 0:
        return "invalid"

    aspect_ratio = height / width  # 비율 계산

    if aspect_ratio > standing_thresh:
        return "standing"
    elif aspect_ratio < lying_thresh:
        return "lying"
    else:
        return "ambiguous"

def fall_detection_fsm(posture, motion_score, state, counter,
                       motion_thresh=0.01, lying_duration_thresh=30):
    is_fall = False

    if posture == "lying" and motion_score < motion_thresh:
        counter += 1
        if counter >= lying_duration_thresh:
            state = STATE_POSSIBLE_FALL
            is_fall = True
    else:
        state = STATE_NORMAL
        counter = 0

    return state, counter, is_fall

def process_single_frame(image, frame_idx):

    # 사람 박스 추출
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    boxes = results.xyxy[0].cpu().numpy()

    # bbox추출 안되면 다음 프레임으로 넘어감
    if len(boxes) == 0:
        log_status_every_3s(frame_idx, detected=False, person_id=None)
        return

    # 관절 추출
    for pid, det in enumerate(boxes):
        if int(det[5]) != 0:
            continue

        x1, y1, x2, y2, _, cls = map(int, det[:6])
        bbox = [x1, y1, x2, y2]
        crop = image_rgb[y1:y2, x1:x2]
        res = pose.process(crop)

        if res.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.visibility] for lm in res.pose_landmarks.landmark])
            # 위에서 정의한 20가지 관절 추출
            joints = extract_20joints_from_landmarks(landmarks, mediapipe_index)
            log_status_every_3s(frame_idx, True, pid)
        else:
            # 관절 실패 → bbox만 존재
            # bbox로 사람은 추출됐는데 관절은 추출 안된 경우 예외처리.
            # bbox로 사람 추출되는 것이 관절 추출되는 것보다 많아서 bbox를 살리기 위해 쓰임
            joints = [-1.0] * (N_JOINTS * 3)
            log_status_every_3s(frame_idx, True, pid)

        # 관절 저장
        joints_np = np.array(joints).reshape(N_JOINTS, 3)

        # 최초 감지자 초기화
        if pid not in person_state:
            person_state[pid] = {'state': STATE_NORMAL, 'counter': 0, 'prev_bbox': bbox}
            person_prev_joints[pid] = None
            person_joint_buffer[pid] = []

        # 모션스코어 계산 (관절 기반)
        if person_prev_joints[pid] is not None and (joints_np[:, 0] != -1.0).all():
            diff = np.linalg.norm(joints_np[:, :2] - person_prev_joints[pid][:, :2], axis=1)
            score = float(np.mean(diff))
        else:
            score = 0.0
        person_prev_joints[pid] = joints_np

        # 자세 추정 및 낙상 FSM
        posture = estimate_posture_from_bbox(bbox)
        state, counter, is_fall = fall_detection_fsm(posture, score, person_state[pid]['state'], person_state[pid]['counter'])
        person_state[pid]['state'] = state
        person_state[pid]['counter'] = counter
        person_state[pid]['prev_bbox'] = bbox

        if is_fall:
            alert_buzzer_and_led()
            print(f"⚠️ 낙상 의심: Frame {frame_idx}, ID={pid}, posture={posture}, motion_score={score:.4f}")

        # AutoEncoder 이상 동작 감지 (동적 상태에서만)
        static_flag = 0 if score < 0.02 else 1
        if static_flag == 1 and (joints_np[:, 0] != -1.0).all():
            flat = joints_np.flatten()   # (20 x 3) → (60,)
            person_joint_buffer[pid].append(flat)
            if len(person_joint_buffer[pid]) > SEQ_LEN:
                person_joint_buffer[pid].pop(0)
            if len(person_joint_buffer[pid]) == SEQ_LEN:
                input_seq = torch.tensor(person_joint_buffer[pid], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    recon = model(input_seq)
                    recon_error = torch.mean((input_seq - recon) ** 2).item()
                print(f"Frame {frame_idx}: ID={pid} 동적상태 → 복원오차 MSE = {recon_error:.5f}")
                if recon_error > 0.01:  # 임계값
                    alert_buzzer_and_led()
                    print(f"이상 동작 감지 (autoencoder): ID={pid}, MSE = {recon_error:.5f}")

#------------------ 프레임 추출 ------------------
#cap = cv2.VideoCapture(0)# 실시간 카메라 입력
#fps = cap.get(cv2.CAP_PROP_FPS)
#interval = int(round(fps / 30))  # 30FPS 기준으로 저장 간격 계산

#frames = []
#frame_ids = []
#count = 0
#while cap.isOpened():
#    ret, frame = cap.read()
#    if not ret:
#        break
#    if count % interval == 0:
#        resized = cv2.resize(frame, (640, 480))
#        frames.append(resized)
#        frame_ids.append(count)
#    count += 1
#cap.release()
# ffmpeg 명령어 (libcamera로 실시간 영상 읽어서 OpenCV에 넘김)


ffmpeg_cmd = [
    "ffmpeg",
    "-f", "v4l2",              # Video4Linux2
    "-framerate", "30",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-i", "/dev/video0",       # 라즈베리파이 카메라 장치 (libcamera-vid가 /dev/video0으로 라우팅됨)
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "pipe:1"
]

# ffmpeg subprocess 실행
def main():
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = []
    frame_ids = []
    count = 0
    interval = 1  # 30FPS 기준 전제 하에 매 프레임 추출

    try:
        while True:
            # 한 프레임 읽기
            raw_frame = proc.stdout.read(WIDTH * HEIGHT * 3)
            if not raw_frame:
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))

            if count % interval == 0:
                resized = cv2.resize(frame, (640, 480))
                frames.append(resized)
                frame_ids.append(count)
                process_single_frame(resized, count)

            count += 1

            # 프리뷰 띄우고 'q' 키 누르면 종료
            cv2.imshow("📷 Live Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("사용자에 의해 종료됨.")

    finally:
        proc.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()