# face_gate门禁 - 最终零报错版
import os
import json
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import time
from datetime import datetime
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保 pet_recognition 模块可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

# 强制屏蔽所有警告（libpng警告彻底消失）
warnings.filterwarnings("ignore")
cv2.setLogLevel(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

import animation
from pet_recognition.scripts.pet_detector import PetDetector
from liveness import LivenessDetector

# ================== 配置 ==================
BASE_PATH = str(Path(__file__).resolve().parent.parent)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_NAME = 'face_gate门禁'
DETECT_EVERY_N = 5
LOG_CD = 20
DOOR_CD = 5
LOG_FILE = os.path.join(BASE_PATH, 'access_log.txt')

# 加载配置
with open(os.path.join(BASE_PATH, 'config.json'), encoding='utf-8') as f:
    config = json.load(f)
THRESHOLD = config['threshold']

# 加载人脸特征
if 'users' in config:
    identities = [u['name'] for u in config['users']]
    identity_anchors = {u['name']: np.array(u['embedding']) for u in config['users']}
else:
    identities = config['identities']
    npz_path = os.path.join(BASE_PATH, 'identity_anchors.npz')
    anchors = np.load(npz_path)
    identity_anchors = {n: anchors[n] for n in identities}

# 模型初始化
mtcnn = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.6,0.7,0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# 摄像头（DSHOW 后端，先于 Tkinter 初始化，避免事件循环冲突）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ 无法打开摄像头！请检查是否被其他应用占用，或更换索引（0→1）")
    exit(1)

# 启动动画窗口（摄像头就绪后再创建 Tk 窗口）
animation.door_start()

# 宠物检测器初始化
PET_CONFIG = os.path.join(BASE_PATH, 'pet_recognition', 'config', 'pet_config.json')
pet_detector = None
PET_ENABLED = os.path.exists(PET_CONFIG)
if PET_ENABLED:
    with open(PET_CONFIG, 'r', encoding='utf-8') as _f:
        _pet_cfg = json.load(_f)
    if os.path.exists(_pet_cfg['model_path']):
        pet_detector = PetDetector(PET_CONFIG)
        PET_INTERVAL = _pet_cfg.get('detect_interval', 10)
        print(f"[宠物检测] 已启用 | 白名单: {_pet_cfg['allowlist']}")
    else:
        print(f"[宠物检测] 模型文件缺失，跳过")

# 活体检测
liveness = LivenessDetector(collect_frames=15, threshold=3.0)
liveness_pending = None  # 当前正在进行活体检测的身份名

# 冷却系统
last_log = {}
last_door = {}

# 日志函数
def write_log(identity, status):
    now = time.time()
    key = "陌生人" if status == "陌生人" else identity
    if key in last_log and now - last_log[key] < LOG_CD:
        return
    last_log[key] = now
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = f"[{t}] 身份：{identity} | 状态：{status}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log)
    print(log.strip())

# 开门冷却函数
def trigger_door_open(name):
    now = time.time()
    if name in last_door and now - last_door[name] < DOOR_CD:
        return
    last_door[name] = now
    animation.door_open(name)

# ================== 主循环 ==================
frame_cnt = 0
last_res = []
last_pet_res = []

print("系统启动成功！按 Q / ESC 退出")
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue
    
    frame = cv2.flip(frame, 1)
    frame_cnt += 1

    # 人脸检测
    if frame_cnt % DETECT_EVERY_N == 0:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        boxes, _ = mtcnn.detect(small)
        last_res = []

        if boxes is not None and len(boxes) > 0:
            last_pet_res = []   # 有人脸时清空宠物结果
            boxes *= 2
            for box in boxes:
                x1,y1,x2,y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        face_tensor = mtcnn(face_rgb)
                        if face_tensor is None or len(face_tensor) == 0:
                            continue

                        emb = resnet(face_tensor.to(DEVICE)).cpu().numpy().squeeze()
                        if len(emb.shape) == 0:
                            emb = emb.reshape(1)

                    best_sim = -1
                    best_name = "陌生人"
                    for n, v in identity_anchors.items():
                        sim = np.dot(emb, v) / (np.linalg.norm(emb)*np.linalg.norm(v))
                        if sim > best_sim:
                            best_sim = sim
                            best_name = n

                    granted = best_sim >= THRESHOLD
                    last_res.append((x1,y1,x2,y2, best_name, best_sim, granted))

                except Exception:
                    continue

    # ── 分支 B：宠物检测（无人脸时触发）──
    if PET_ENABLED and pet_detector is not None and len(last_res) == 0:
        if frame_cnt % PET_INTERVAL == 0:
            last_pet_res = []
            pet_result = pet_detector.detect(frame)
            if pet_result is not None:
                breed_zh, conf, breed_en, box, granted = pet_result
                x1, y1, x2, y2 = box
                last_pet_res.append((x1, y1, x2, y2, breed_en, conf, granted))

    # 绘制识别结果（人脸）
    for x1,y1,x2,y2,name,sim,granted in last_res:
        if granted:
            # ── 活体检测 ──
            if liveness_pending != name:
                liveness.start()
                liveness_pending = name

            liveness_passed, liveness_progress = liveness.check((x1, y1, x2, y2))

            if liveness_passed:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{name} {sim:.2f} LIVE",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                trigger_door_open(name)
                write_log(name, "通过")
                liveness_pending = None
            else:
                # 采集中 → 显示进度
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)
                cv2.putText(frame,f"{name} {sim:.2f} liveness:{int(liveness_progress*15)}/15",
                            (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,200,0),2)
        else:
            liveness.reset()
            liveness_pending = None
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"Stranger {sim:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            animation.door_show_fail()
            write_log("陌生人", "陌生人")

    # 绘制识别结果（宠物）
    for x1,y1,x2,y2,name,conf,granted in last_pet_res:
        if granted:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{name} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            trigger_door_open(name)
            write_log(name, f"通过 [品种: {name}]")
        else:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"DENY {name} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            write_log(name, f"拒绝 [品种: {name}]")

    # 窗口显示
    cv2.imshow(WINDOW_NAME, frame)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # 窗口尚未就绪时忽略
    animation.door_update()

    # 退出按键
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
animation.destroy_all()
print("系统安全退出")