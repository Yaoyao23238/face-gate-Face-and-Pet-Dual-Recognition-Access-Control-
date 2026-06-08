import os
import json
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# ===================== 配置（和训练/检测脚本保持一致）=====================
BASE_PATH = str(Path(__file__).resolve().parent.parent)
TEST_DIR = os.path.join(BASE_PATH, "test")  # 测试集目录
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SUFFIX = ('.png', '.jpg', '.jpeg', '.bmp')

# 中文路径读图函数（复用训练脚本）
def load_image_cv2_safe(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# ===================== 加载模型 & 训练配置 =====================
# 加载阈值、身份列表
with open(os.path.join(BASE_PATH, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

# 加载人脸锚点特征
npz_path = os.path.join(BASE_PATH, "identity_anchors.npz")
anchors = np.load(npz_path)
identities = config["identities"]
identity_anchors = {name: anchors[name] for name in identities}

# 初始化模型（和训练完全一致）
mtcnn = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# 余弦相似度计算
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ===================== 开始统计指标 =====================
# 统计变量
total_sample = 0
correct = 0
TP = 0   # 本人识别通过
FN = 0   # 本人识别拒绝
TN = 0   # 陌生人识别拒绝
FP = 0   # 陌生人识别通过

print("="*60)
print("开始face_gate模型准确率评测...")
print(f"识别阈值: {THRESHOLD:.4f}")
print(f"已注册身份: {identities}")
print("="*60)

# 遍历测试集所有文件夹
for person_name in os.listdir(TEST_DIR):
    person_test_dir = os.path.join(TEST_DIR, person_name)
    if not os.path.isdir(person_test_dir):
        continue

    # 判断当前样本是【已注册本人】还是【陌生人】
    is_registered = person_name in identities
    print(f"\n正在测试身份: {person_name} | 是否为注册人员: {is_registered}")

    for img_name in os.listdir(person_test_dir):
        if not img_name.lower().endswith(IMAGE_SUFFIX):
            continue
        
        img_path = os.path.join(person_test_dir, img_name)
        img = load_image_cv2_safe(img_path)
        if img is None:
            print(f"  跳过无效图片: {img_name}")
            continue

        total_sample += 1
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(img_rgb)

        # 未检测到人脸，直接判定为识别失败
        if face_tensor is None or len(face_tensor) == 0:
            print(f"  {img_name} : 未检测到人脸")
            if is_registered:
                FN += 1
            else:
                TN += 1
            continue

        # 只取第一张检测到的人脸（合影等多脸情况取最前面那张）
        if len(face_tensor) > 1:
            face_tensor = face_tensor[0:1]

        # 提取人脸特征
        with torch.no_grad():
            emb = resnet(face_tensor.to(DEVICE)).cpu().numpy().squeeze()

        # 匹配最优身份
        best_sim = -1
        best_match = ""
        for name, anchor_vec in identity_anchors.items():
            sim = cosine_similarity(emb, anchor_vec)
            if sim > best_sim:
                best_sim = sim
                best_match = name

        # 根据阈值判断是否授权通过
        predict_pass = (best_sim >= THRESHOLD)

        # 统计指标
        if is_registered:
            # 真实标签：本人应该匹配到自己
            if predict_pass and best_match == person_name:
                TP += 1
                correct += 1
                print(f"  {img_name} ✅ 正确 | 相似度:{best_sim:.4f} | 匹配:{best_match}")
            elif predict_pass and best_match != person_name:
                FN += 1
                print(f"  {img_name} ❌ 误配 | 相似度:{best_sim:.4f} | 匹配到:{best_match}（应是:{person_name}）")
            else:
                FN += 1
                print(f"  {img_name} ❌ 漏检 | 相似度:{best_sim:.4f}")
        else:
            # 真实标签：应该拒绝
            if not predict_pass:
                TN += 1
                correct += 1
                print(f"  {img_name} ✅ 正确 | 相似度:{best_sim:.4f}")
            else:
                FP += 1
                print(f"  {img_name} ❌ 误判 | 相似度:{best_sim:.4f} | 匹配:{best_match}")

# ===================== 输出最终评测结果 =====================
print("\n" + "="*60)
print("📊 模型评测最终报告")
print("="*60)
print(f"总测试样本数: {total_sample}")
print(f"预测正确样本数: {correct}")
accuracy = (correct / total_sample) * 100 if total_sample > 0 else 0
print(f"✅ 整体准确率: {accuracy:.2f} %")
print("-"*60)
print(f"TP(本人正常通过): {TP}")
print(f"FN(本人被拒绝/漏检): {FN}")
print(f"TN(陌生人正常拒绝): {TN}")
print(f"FP(陌生人误判通过): {FP}")
print("="*60)