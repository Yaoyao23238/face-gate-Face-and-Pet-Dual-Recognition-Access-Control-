# evaluate_robustness.py — 人脸识别鲁棒性测试
# 对测试集施加多种扰动，评估模型在干扰下的准确率衰减

import os, json, sys, warnings
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

warnings.filterwarnings("ignore")
cv2.setLogLevel(0)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ==================== 配置 ====================
BASE_PATH = os.path.join(os.path.dirname(__file__), '..')
TEST_DIR = os.path.join(BASE_PATH, "test")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SUFFIX = ('.png', '.jpg', '.jpeg', '.bmp')

# ==================== 加载模型 & 配置 ====================
print("[1/4] 加载模型...")
with open(os.path.join(BASE_PATH, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

anchors = np.load(os.path.join(BASE_PATH, "identity_anchors.npz"))
identities = config["identities"]
identity_anchors = {name: anchors[name] for name in identities}

mtcnn = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def extract_embedding(img):
    """提取单张图片的人脸特征，无脸返回 None"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    if face_tensor is None or len(face_tensor) == 0:
        return None
    if len(face_tensor) > 1:
        face_tensor = face_tensor[0:1]
    with torch.no_grad():
        emb = resnet(face_tensor.to(DEVICE)).cpu().numpy().squeeze()
    return emb


def classify(emb):
    """给定 embedding，返回 (matched_name, similarity, is_pass)"""
    best_sim, best_name = -1, ""
    for n, v in identity_anchors.items():
        s = cos_sim(emb, v)
        if s > best_sim:
            best_sim, best_name = s, n
    return best_name, float(best_sim), best_sim >= THRESHOLD


# ==================== 扰动函数 ====================
def perturb_brightness(img, factor):
    """factor: 0.5=变暗一半, 1.5=变亮50%"""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def perturb_contrast(img, factor):
    """factor: 0.5=低对比度, 1.5=高对比度"""
    gray = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img.astype(np.float32) - gray) * factor + gray, 0, 255).astype(np.uint8)


def perturb_blur(img, ksize):
    """ksize: 高斯核大小，必须是奇数"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def perturb_noise(img, density):
    """density: 椒盐噪声比例"""
    out = img.copy()
    h, w = out.shape[:2]
    n_pixels = int(h * w * density)
    for _ in range(n_pixels // 2):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        out[y, x] = [255, 255, 255]  # salt
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        out[y, x] = [0, 0, 0]        # pepper
    return out


def perturb_rotate(img, angle):
    """angle: 旋转角度（度）"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ==================== 扰动配置表 ====================
PERTURBATIONS = [
    # (名称, 函数, 参数列表)
    ("亮度×0.5",      perturb_brightness, [0.5]),
    ("亮度×0.7",      perturb_brightness, [0.7]),
    ("亮度×1.3",      perturb_brightness, [1.3]),
    ("亮度×1.5",      perturb_brightness, [1.5]),
    ("对比度×0.5",    perturb_contrast,   [0.5]),
    ("对比度×0.7",    perturb_contrast,   [0.7]),
    ("对比度×1.3",    perturb_contrast,   [1.3]),
    ("高斯模糊 k=3",   perturb_blur,       [3]),
    ("高斯模糊 k=5",   perturb_blur,       [5]),
    ("椒盐噪声 1%",    perturb_noise,      [0.01]),
    ("椒盐噪声 3%",    perturb_noise,      [0.03]),
    ("旋转 +5°",      perturb_rotate,     [5]),
    ("旋转 -5°",      perturb_rotate,     [-5]),
    ("旋转 +10°",     perturb_rotate,     [10]),
]

# ==================== 评估函数 ====================
def evaluate_on_images(images, label_name, is_registered):
    """
    对一组图片做评估，返回准确率
    - is_registered: 期望通过（本人）还是拒绝（陌生人）
    """
    if len(images) == 0:
        return 0.0
    correct = 0
    for img in images:
        emb = extract_embedding(img)
        if emb is None:
            if not is_registered:
                correct += 1   # 陌生人未检测到脸 → 算正确（拒绝）
            continue
        _, _, passed = classify(emb)
        if passed == is_registered:
            correct += 1
    return correct / len(images)


def collect_test_images():
    """收集测试集：返回 {person_name: [images]} 和陌生人列表"""
    registered_images = {name: [] for name in identities}
    stranger_images = []

    if not os.path.exists(TEST_DIR):
        print(f"  ⚠ 测试目录不存在: {TEST_DIR}")
        return registered_images, stranger_images

    for person_name in os.listdir(TEST_DIR):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith(IMAGE_SUFFIX):
                continue
            img = load_image(os.path.join(person_dir, fname))
            if img is None:
                continue
            if person_name in registered_images:
                registered_images[person_name].append(img)
            else:
                stranger_images.append(img)

    return registered_images, stranger_images


# ==================== 主流程 ====================
def main():
    print(f"\n[2/4] 收集测试集 (路径: {TEST_DIR})...")
    reg_imgs, str_imgs = collect_test_images()

    total_reg = sum(len(v) for v in reg_imgs.values())
    total_str = len(str_imgs)
    print(f"  注册用户样本: {total_reg} 张 (分布在 {len(reg_imgs)} 人)")
    print(f"  陌生人样本:   {total_str} 张")

    if total_reg == 0 and total_str == 0:
        print("  ❌ 测试集为空，无法评估")
        return

    # 1) 基线准确率
    print(f"\n[3/4] 评估基线准确率 (无扰动)...")
    baseline_correct = 0
    baseline_total = 0
    for name, imgs in reg_imgs.items():
        for img in imgs:
            emb = extract_embedding(img)
            if emb is None: continue
            matched, _, passed = classify(emb)
            baseline_total += 1
            if passed and matched == name:
                baseline_correct += 1
    for img in str_imgs:
        emb = extract_embedding(img)
        if emb is None:
            baseline_correct += 1; baseline_total += 1; continue
        _, _, passed = classify(emb)
        baseline_total += 1
        if not passed:
            baseline_correct += 1
    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0
    print(f"  基线准确率: {baseline_acc:.2f}% ({baseline_correct}/{baseline_total})")

    # 2) 扰动测试
    print(f"\n[4/4] 扰动鲁棒性测试 ({len(PERTURBATIONS)} 种扰动)...")
    print(f"\n{'扰动类型':<20} {'准确率':>8} {'下降':>8} {'评级':>6}")
    print("-" * 48)

    scores = []
    for pname, pfunc, pargs in PERTURBATIONS:
        correct = 0
        total = 0
        for name, imgs in reg_imgs.items():
            for img in imgs:
                try:
                    pimg = pfunc(img, *pargs)
                    emb = extract_embedding(pimg)
                    if emb is None: continue
                    matched, _, passed = classify(emb)
                    total += 1
                    if passed and matched == name:
                        correct += 1
                except Exception:
                    continue
        for img in str_imgs:
            try:
                pimg = pfunc(img, *pargs)
                emb = extract_embedding(pimg)
                if emb is None:
                    correct += 1; total += 1; continue
                _, _, passed = classify(emb)
                total += 1
                if not passed:
                    correct += 1
            except Exception:
                continue

        acc = correct / total * 100 if total > 0 else 0
        drop = baseline_acc - acc
        if drop < 5:       grade = "🟢优"
        elif drop < 15:    grade = "🟡良"
        elif drop < 30:    grade = "🟠中"
        else:              grade = "🔴差"

        print(f"{pname:<20} {acc:>7.2f}% {drop:>7.1f}% {grade:>6}")
        scores.append(acc)

    # 3) 综合鲁棒性分数
    avg_perturb = np.mean(scores) if scores else 0
    robustness = (avg_perturb / baseline_acc * 100) if baseline_acc > 0 else 0

    print(f"\n{'='*50}")
    print(f"  基线准确率:      {baseline_acc:.2f}%")
    print(f"  扰动平均准确率:  {avg_perturb:.2f}%")
    print(f"  鲁棒性分数:      {robustness:.1f}/100")
    print(f"  阈值:            {THRESHOLD:.4f}")
    print(f"{'='*50}")

    if robustness >= 90:
        print("  ✅ 鲁棒性优秀 — 模型对各种干扰抵抗能力强")
    elif robustness >= 80:
        print("  🟢 鲁棒性良好 — 轻微扰动影响不大")
    elif robustness >= 70:
        print("  🟡 鲁棒性一般 — 建议优化")
    else:
        print("  🔴 鲁棒性较差 — 必须优化")

    # 优化建议
    if robustness < 80:
        print(f"\n  📋 优化建议:")
        print(f"  1. 降低阈值: 当前={THRESHOLD:.4f}，降到 {THRESHOLD-0.05:.4f}~{THRESHOLD-0.10:.4f} 可提升通过率")
        print(f"  2. 增加注册照片: 每人在不同光照/角度下多拍 5-10 张")
        print(f"  3. 提高注册照片质量: 确保每张都能提取到清晰人脸")
        print(f"  4. 在训练时用 cv2 做数据增强 (亮度/对比度/模糊)")
        print(f"  5. 使用 FaceNet 的 L2 归一化模式 (当前未启用)")


if __name__ == '__main__':
    main()
