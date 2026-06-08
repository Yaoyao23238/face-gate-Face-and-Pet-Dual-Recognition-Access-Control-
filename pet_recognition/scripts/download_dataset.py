# download_dataset.py
# 通过 torchvision 下载 Oxford-IIIT Pet -> YOLO 格式标注 -> 拆分为 train/val/test (70/15/15)
import os, sys, shutil, re
from pathlib import Path
from xml.etree import ElementTree as ET
import random
from torchvision.datasets import OxfordIIITPet

# --- 路径配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW   = PROJECT_ROOT / 'pet_recognition' / 'data' / 'raw'
DATA_OUT   = PROJECT_ROOT / 'pet_recognition' / 'data'
TV_ROOT    = DATA_RAW

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# --- 37 品种严格顺序（class_id = 索引）---
BREED_NAMES = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx',
    'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound',
    'Beagle', 'Boxer', 'Chihuahua', 'English_Cocker_Spaniel',
    'English_Setter', 'German_Shorthaired', 'Great_Pyrenees',
    'Havanese', 'Japanese_Chin', 'Keeshond', 'Leonberger',
    'Miniature_Pinscher', 'Newfoundland', 'Pomeranian', 'Pug',
    'Saint_Bernard', 'Samoyed', 'Scottish_Terrier', 'Shiba_Inu',
    'Staffordshire_Bull_Terrier', 'Wheaten_Terrier', 'Yorkshire_Terrier',
]
BREED_TO_ID = {name.lower(): i for i, name in enumerate(BREED_NAMES)}


# --- 标注解析 ---
def parse_xml(xml_path):
    """解析 Pascal VOC XML，返回 (filename, w, h, objects_list)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        objects.append({
            'xmin': int(float(bndbox.find('xmin').text)),
            'ymin': int(float(bndbox.find('ymin').text)),
            'xmax': int(float(bndbox.find('xmax').text)),
            'ymax': int(float(bndbox.find('ymax').text)),
        })
    return filename, w, h, objects


# --- 主流程 ---
def main():
    print("=" * 60)
    print("Oxford-IIIT Pet -> YOLO format dataset preparation")
    print("=" * 60)

    # 1. 通过 torchvision 下载
    print("\n[1/4] Downloading via torchvision...")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    try:
        OxfordIIITPet(root=str(TV_ROOT), split='trainval', download=True)
    except RuntimeError as e:
        if 'already exists' in str(e):
            print("  Dataset already exists, skipping download")
        else:
            raise

    TV_DIR = TV_ROOT / 'oxford-iiit-pet'
    IMAGES_DIR = TV_DIR / 'images'
    XML_DIR    = TV_DIR / 'annotations' / 'xmls'
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Annotations: {XML_DIR}")

    # 2. 转换 YOLO 标注
    print("\n[2/4] Converting to YOLO format...")
    xml_files = sorted(XML_DIR.glob('*.xml'))
    records = []
    skipped = 0

    for xml_path in xml_files:
        try:
            filename, w, h, objects = parse_xml(xml_path)
        except Exception:
            skipped += 1
            continue

        stem = Path(filename).stem
        # 从文件名提取品种（'Abyssinian_1' -> 'Abyssinian'，大小写不敏感）
        breed = re.sub(r'_\d+$', '', stem).lower()
        if breed not in BREED_TO_ID:
            continue

        cid = BREED_TO_ID[breed]
        lines = []
        for obj in objects:
            cx = (obj['xmin'] + obj['xmax']) / 2.0 / w
            cy = (obj['ymin'] + obj['ymax']) / 2.0 / h
            bw = (obj['xmax'] - obj['xmin']) / w
            bh = (obj['ymax'] - obj['ymin']) / h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if lines:
            records.append((stem, lines, filename))

    if skipped:
        print(f"  Skipped {skipped} unparseable XMLs")
    print(f"  Converted {len(records)} images")

    # 3. 拆分并拷贝
    print("\n[3/4] Splitting dataset (70/15/15)...")
    rng = random.Random(SEED)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        'train': records[:n_train],
        'val':   records[n_train:n_train + n_val],
        'test':  records[n_train + n_val:],
    }

    for split_name, split_records in splits.items():
        img_out = DATA_OUT / split_name / 'images'
        lbl_out = DATA_OUT / split_name / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem, lines, orig_filename in split_records:
            src_img = IMAGES_DIR / orig_filename
            dst_img = img_out / orig_filename
            if src_img.exists():
                try:
                    shutil.copy2(src_img, dst_img)
                except OSError:
                    pass
            else:
                alt = IMAGES_DIR / f"{stem}.jpg"
                if alt.exists():
                    try:
                        shutil.copy2(alt, dst_img)
                    except OSError:
                        pass

            lbl_path = lbl_out / f"{stem}.txt"
            with open(lbl_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        print(f"  {split_name}: {len(split_records)} images")

    print(f"\n[4/4] Done! Total {n} images split into train/val/test")
    print(f"  Train: {n_train}  Val: {n_val}  Test: {n - n_train - n_val}")


if __name__ == '__main__':
    main()
