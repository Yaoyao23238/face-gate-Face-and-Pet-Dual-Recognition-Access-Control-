import os
import json
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

# ===================== 配置参数（你的路径）=====================
data_dir = str(Path(__file__).resolve().parent.parent)
output_dir = data_dir
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================== 初始化模型（纯CPU）=====================
print("正在加载MTCNN+InceptionResnetV1模型...")
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ===================== 修复：中文路径100%兼容读取（彻底解决乱码）=====================
def load_image_cv2_safe(path):
    # 强制使用 np.fromfile + imdecode，抛弃 cv2.imread，彻底解决中文乱码
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# ===================== 加载并处理所有训练图像 =====================
embeddings = []
labels = []  # 存储身份标签
print(f"开始读取文件夹：{data_dir} 中的人脸图片")

# 递归扫描 knowns 子目录
knowns_dir = os.path.join(data_dir, 'knowns')
if not os.path.exists(knowns_dir):
    print(f"错误：找不到 knowns 目录 {knowns_dir}")
    exit()

for person_name in os.listdir(knowns_dir):
    person_dir = os.path.join(knowns_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    
    print(f"处理身份：{person_name}")
    person_embeddings = []
    
    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(person_dir, filename)
            # 读取图片
            img = load_image_cv2_safe(img_path)
            if img is None:
                print(f"跳过无效图片：{filename}")
                continue
            
            # 格式转换
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # 人脸检测+对齐
            face_tensor = mtcnn(img_pil)
            if face_tensor is not None and len(face_tensor) > 0:
                with torch.no_grad():
                    embedding = resnet(face_tensor.to(device))
                person_embeddings.append(embedding.detach().cpu().numpy())
                print(f"  {filename} ✓")
    
    if person_embeddings:
        # 计算该身份的平均锚点
        person_anchor = np.mean(np.concatenate(person_embeddings, axis=0), axis=0)
        embeddings.append(person_anchor)
        labels.append(person_name)
        print(f"  提取到 {len(person_embeddings)} 张照片，锚点已保存")

# 合并所有嵌入向量（每个身份一个锚点）
if len(embeddings) == 0:
    print("错误：未提取到任何人脸特征！")
    exit()
    
embeddings = np.array(embeddings)
print(f"\n共 {len(labels)} 个身份，{embeddings.shape[0]} 个锚点")

# ===================== 计算类间余弦相似度（跨身份最小相似度作为阈值）=====================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 所有已注册身份内部的两两相似度
all_intra_sims = []
identities_anchor = {}  # {person_name: anchor_array}

for i, label in enumerate(labels):
    identities_anchor[label] = embeddings[i]
    # 重新计算该身份的内部相似度（从原图片提取所有embedding）
    person_dir = os.path.join(knowns_dir, label)
    person_embs = []
    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(person_dir, filename)
            img = load_image_cv2_safe(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            face_tensor = mtcnn(img_pil)
            if face_tensor is not None and len(face_tensor) > 0:
                with torch.no_grad():
                    if len(face_tensor) > 1:
                        face_tensor = face_tensor[0:1]  # 多脸时只取第一张
                    emb = resnet(face_tensor.to(device)).detach().cpu().numpy().squeeze()
                person_embs.append(emb)
    
    # 该身份内部相似度
    for a in range(len(person_embs)):
        for b in range(a+1, len(person_embs)):
            all_intra_sims.append(cosine_similarity(person_embs[a], person_embs[b]))

all_intra_sims = np.array(all_intra_sims) if all_intra_sims else np.array([0.5])
stats = {
    'min': float(np.min(all_intra_sims)),
    'max': float(np.max(all_intra_sims)),
    'mean': float(np.mean(all_intra_sims)),
    'std': float(np.std(all_intra_sims)),
    'recommended_threshold': float(np.mean(all_intra_sims) - 2.0 * np.std(all_intra_sims))
}

# ===================== 保存结果（保留已有阈值，不自动覆盖）=====================
np.savez(os.path.join(output_dir, 'identity_anchors.npz'), **identities_anchor)

config_path = os.path.join(output_dir, 'config.json')
existing_threshold = None
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        existing_threshold = json.load(f).get('threshold')

threshold_out = existing_threshold if existing_threshold is not None else stats['recommended_threshold']
keep_note = f'  保留已有阈值: {threshold_out:.4f}' if existing_threshold else f'  使用推荐阈值: {threshold_out:.4f}'

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump({
        'identities': labels,
        'threshold': threshold_out,
        'similarity_stats': stats
    }, f, indent=4)

print(keep_note)

# ===================== 打印统计信息 =====================
print("\n" + "="*50)
print("✅ 训练完成！")
print(f"注册身份: {labels}")
print(f"锚点数量: {embeddings.shape[0]}")
print(f"向量维度: {embeddings.shape[1]}")
print("\n📊 类内相似度分析：")
print(f"最小相似度: {stats['min']:.4f}")
print(f"最大相似度: {stats['max']:.4f}")
print(f"平均相似度: {stats['mean']:.4f}")
print(f"🎯 推荐识别阈值: {stats['recommended_threshold']:.4f}")
print("="*50)