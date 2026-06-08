# web_gate.py — Face Gate Web UI (Flask + SocketIO 版)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os, json, time, threading, warnings
from datetime import datetime
import cv2, numpy as np, torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, Response, request, jsonify
from g2.liveness import LivenessDetector

warnings.filterwarnings("ignore")
cv2.setLogLevel(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==================== 配置 ====================
BASE_PATH = Path(__file__).resolve().parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DETECT_EVERY_N, LOG_CD, DOOR_CD, HTTP_PORT = 5, 20, 5, 5000

# Flask
app = Flask(__name__)
TEMPLATE_HTML = (BASE_PATH / 'templates' / 'index.html').read_text(encoding='utf-8')

# ==================== 模型初始化 ====================
print("[Init] Loading face models...")
mtcnn = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# 宠物
pet_detector = None; PET_ENABLED = False; PET_INTERVAL = 10
pet_config_path = BASE_PATH / 'pet_recognition' / 'config' / 'pet_config.json'
all_breeds = []
if pet_config_path.exists():
    with open(pet_config_path, encoding='utf-8') as f:
        pet_cfg = json.load(f)
    all_breeds = sorted(pet_cfg.get('en_to_zh', {}).keys())
    if os.path.exists(pet_cfg['model_path']):
        from pet_recognition.scripts.pet_detector import PetDetector
        pet_detector = PetDetector(str(pet_config_path)); PET_ENABLED = True
        PET_INTERVAL = pet_cfg.get('detect_interval', 10)
        print(f"[Init] Pet detector loaded, allowlist: {pet_cfg['allowlist']}")

# 摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened(): print("ERROR: Cannot open camera"); exit(1)

# ==================== 人脸配置 ====================
def reload_configs():
    global identity_anchors, identities, THRESHOLD, pet_cfg
    with open(BASE_PATH / 'config.json', encoding='utf-8') as f:
        c = json.load(f)
    THRESHOLD = c['threshold']; identities = list(c['identities'])
    a = np.load(BASE_PATH / 'identity_anchors.npz')
    identity_anchors = {n: a[n] for n in identities}
    if pet_config_path.exists():
        with open(pet_config_path, encoding='utf-8') as f:
            pet_cfg = json.load(f)

reload_configs()

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ==================== 共享状态 ====================
lock = threading.Lock()
shared_frame = None
status = {
    'gate_state': 'idle', 'gate_name': '', 'detections': [],
    'pass_count': 0, 'deny_count': 0, 'log': [], 'fps': 0, 'timestamp': '',
    'registered_faces': identities, 'pet_allowlist': pet_cfg.get('allowlist', []),
    'all_breeds': all_breeds,
}

liveness = LivenessDetector(collect_frames=15, threshold=3.0)
liveness_pending = None
shutdown_flag = threading.Event()

def draw_detections(frame, dets):
    for d in dets:
        x1,y1,x2,y2 = d['box']; color = (0,255,0) if d['granted'] else (0,0,255)
        label = f"{d['name']} {d['confidence']:.2f}" if d['granted'] else f"DENY {d['name']}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
        cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
        cv2.putText(frame,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)
    return frame

# ==================== 检测循环 ====================
def detection_loop():
    global shared_frame, status
    frame_cnt = 0; gate_state = 'idle'; gate_name = ''; gate_time = 0
    last_detections = []; last_log = {}; last_door = {}; fps_times = []

    while not shutdown_flag.is_set():
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        frame = cv2.flip(frame,1); frame_cnt += 1; now = time.time()

        if frame_cnt % DETECT_EVERY_N == 0:
            h,w = frame.shape[:2]
            small = cv2.resize(frame,(w//2,h//2))
            boxes,_ = mtcnn.detect(small)
            last_detections = []

            if boxes is not None and len(boxes) > 0:
                boxes *= 2
                for box in boxes:
                    x1,y1,x2,y2 = map(int,box)
                    x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w,x2),min(h,y2)
                    face = frame[y1:y2,x1:x2]
                    if face.size == 0: continue
                    try:
                        face_rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                        with torch.no_grad():
                            ft = mtcnn(face_rgb)
                            if ft is None or len(ft)==0: continue
                            emb = resnet(ft.to(DEVICE)).cpu().numpy().squeeze()
                            if len(emb.shape)==0: emb = emb.reshape(1)
                        best_sim, best_name = -1, 'Stranger'
                        for n,v in identity_anchors.items():
                            s = cos_sim(emb,v)
                            if s > best_sim: best_sim, best_name = s, n
                        granted = bool(best_sim >= THRESHOLD)
                        display = best_name if granted else 'Stranger'
                        last_detections.append({
                            'type':'face','box':[x1,y1,x2,y2],
                            'name':display,'confidence':round(best_sim,3),'granted':granted,
                        })
                        if granted:
                            global liveness_pending
                            if liveness_pending != best_name:
                                liveness.start(); liveness_pending = best_name
                            lv_passed, lv_prog = liveness.check((x1,y1,x2,y2))
                            if lv_passed:
                                if best_name not in last_log or now-last_log[best_name]>=LOG_CD:
                                    last_log[best_name]=now
                                    with lock:
                                        status['log'].append({'time':datetime.now().strftime('%H:%M:%S'),'name':display,'status':'通过'})
                                        if len(status['log'])>50: status['log'].pop(0)
                                        status['pass_count']+=1
                                if best_name not in last_door or now-last_door[best_name]>=DOOR_CD:
                                    last_door[best_name]=now; gate_state='open'; gate_name=display; gate_time=now
                                liveness_pending = None
                        else:
                            liveness.reset(); liveness_pending = None
                            with lock: status['deny_count']+=1
                            if 'Stranger' not in last_door or now-last_door['Stranger']>=DOOR_CD:
                                last_door['Stranger']=now; gate_state='deny'; gate_name=display; gate_time=now
                    except Exception: continue

            elif PET_ENABLED and pet_detector and frame_cnt % PET_INTERVAL == 0:
                pr = pet_detector.detect(frame)
                if pr is not None:
                    breed_zh,conf,breed_en,box,granted = pr
                    x1,y1,x2,y2 = map(int,box)
                    last_detections.append({
                        'type':'pet','box':[x1,y1,x2,y2],
                        'name':breed_en,'confidence':round(float(conf),3),'granted':granted,
                    })
                    if granted:
                        if breed_en not in last_log or now-last_log[breed_en]>=LOG_CD:
                            last_log[breed_en]=now
                            with lock:
                                status['log'].append({'time':datetime.now().strftime('%H:%M:%S'),'name':breed_en,'status':'通过'})
                                if len(status['log'])>50: status['log'].pop(0)
                                status['pass_count']+=1
                        if breed_en not in last_door or now-last_door[breed_en]>=DOOR_CD:
                            last_door[breed_en]=now; gate_state='open'; gate_name=breed_en; gate_time=now
                    else:
                        with lock: status['deny_count']+=1
                        if breed_en not in last_door or now-last_door[breed_en]>=DOOR_CD:
                            last_door[breed_en]=now; gate_state='deny'; gate_name=breed_en; gate_time=now

        if gate_state=='open' and now-gate_time>3.0: gate_state='idle'
        elif gate_state=='deny' and now-gate_time>3.5: gate_state='idle'

        frame = draw_detections(frame,last_detections)
        _,jpeg = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,70])

        fps_times.append(now)
        if len(fps_times)>30: fps_times.pop(0)
        fps_v = round(len(fps_times)/(fps_times[-1]-fps_times[0]),1) if len(fps_times)>1 else 0

        with lock:
            shared_frame = jpeg.tobytes()
            status['gate_state']=gate_state; status['gate_name']=gate_name
            status['detections']=last_detections; status['fps']=fps_v
            status['timestamp']=datetime.now().strftime('%H:%M:%S')



# ==================== Flask 路由 ====================

@app.route('/')
def index():
    return TEMPLATE_HTML

@app.route('/video_feed')
def video_feed():
    def generate():
        while not shutdown_flag.is_set():
            with lock: fd = shared_frame
            if fd:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fd + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    with lock: s = dict(status)
    s['registered_faces'] = identities
    s['pet_allowlist'] = pet_cfg.get('allowlist', [])
    s['all_breeds'] = all_breeds
    return jsonify(s)

@app.route('/breeds')
def get_breeds():
    bl = []
    for en in all_breeds:
        zh = pet_cfg.get('en_to_zh', {}).get(en, en)
        bl.append({'en': en, 'zh': zh})
    return jsonify({'breeds': bl})

@app.route('/faces')
def get_faces():
    return jsonify({'faces': identities})

# ── 宠物管理 ──
@app.route('/pet/add', methods=['POST'])
def pet_add():
    breed = request.args.get('breed','')
    if not breed: return jsonify({'error':'Missing breed'}), 400
    allowlist = list(pet_cfg.get('allowlist',[]))
    if breed not in allowlist:
        allowlist.append(breed); pet_cfg['allowlist']=allowlist
        with open(pet_config_path,'w',encoding='utf-8') as f: json.dump(pet_cfg,f,indent=2,ensure_ascii=False)
        if pet_detector: pet_detector.allowlist=set(allowlist)
    return jsonify({'result':f'Added: {breed}'})

@app.route('/pet/remove', methods=['POST'])
def pet_remove():
    breed = request.args.get('breed','')
    if not breed: return jsonify({'error':'Missing breed'}), 400
    allowlist = list(pet_cfg.get('allowlist',[]))
    if breed in allowlist:
        allowlist.remove(breed); pet_cfg['allowlist']=allowlist
        with open(pet_config_path,'w',encoding='utf-8') as f: json.dump(pet_cfg,f,indent=2,ensure_ascii=False)
        if pet_detector: pet_detector.allowlist=set(allowlist)
    return jsonify({'result':f'Removed: {breed}'})

# ── 人脸管理 ──
@app.route('/face/register', methods=['POST'])
def face_register():
    name = request.args.get('name','').strip()
    if not name: return jsonify({'error':'Missing name'}), 400
    with lock: fd = shared_frame
    if not fd: return jsonify({'error':'No frame'}), 500
    arr = np.frombuffer(fd,dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if frame_bgr is None: return jsonify({'error':'Decode failed'}), 500
    boxes,_ = mtcnn.detect(frame_bgr)
    if boxes is None or len(boxes)==0: return jsonify({'error':'No face detected'}), 400
    try:
        x1,y1,x2,y2 = map(int,boxes[0])
        face = frame_bgr[max(0,y1):y2,max(0,x1):x2]
        face_rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            ft = mtcnn(face_rgb)
            if ft is None or len(ft)==0: return jsonify({'error':'Alignment failed'}), 400
            emb = resnet(ft.to(DEVICE)).cpu().numpy().squeeze()
        identity_anchors[name]=emb
        if name not in identities: identities.append(name)
        np.savez(BASE_PATH/'identity_anchors.npz',**identity_anchors)
        with open(BASE_PATH/'config.json','w',encoding='utf-8') as fw:
            json.dump({'identities':identities,'threshold':THRESHOLD},fw,indent=2,ensure_ascii=False)
        return jsonify({'result':f'Registered: {name}'})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/face/remove', methods=['POST'])
def face_remove():
    name = request.args.get('name','')
    if not name: return jsonify({'error':'Missing name'}), 400
    if name not in identity_anchors: return jsonify({'error':f'Not found: {name}'}), 400
    del identity_anchors[name]; identities.remove(name)
    np.savez(BASE_PATH/'identity_anchors.npz',**identity_anchors)
    with open(BASE_PATH/'config.json','w',encoding='utf-8') as fw:
        json.dump({'identities':identities,'threshold':THRESHOLD},fw,indent=2,ensure_ascii=False)
    return jsonify({'result':f'Removed: {name}'})

# ── 退出 ──
@app.route('/shutdown', methods=['POST'])
def shutdown():
    def _exit():
        time.sleep(0.5)
        shutdown_flag.set()
        cap.release()
        os._exit(0)
    threading.Thread(target=_exit,daemon=True).start()
    return jsonify({'result':'Shutting down...'})


# ==================== 入口 ====================
if __name__ == '__main__':
    print("[Init] Starting detection thread...")
    threading.Thread(target=detection_loop,daemon=True).start()
    time.sleep(2)
    print(f"[Init] Web server at http://localhost:{HTTP_PORT}")
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False)
