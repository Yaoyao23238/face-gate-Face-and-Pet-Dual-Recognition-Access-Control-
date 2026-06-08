# train.py — YOLOv8m fine-tune on Oxford-IIIT Pet
# 适配 RTX 4050 6GB：batch=8, amp=True
import shutil
from pathlib import Path
import torch
import time as _time
import os as _os

# ── 修复 Windows Defender 实时扫描导致 torch.save 失败 ──
# 原因：写入 136MB 大文件耗时数秒，Defender 在此期间扫描并锁住文件句柄
# 策略：先写入临时文件（.tmp），再用原子 rename 到目标路径，rename 瞬间完成无法被拦截
_orig_torch_save = torch.save

def _safe_torch_save(obj, f, *args, **kwargs):
    # 仅对文件路径生效，file-like object 直接用原版
    if not isinstance(f, (str, _os.PathLike)):
        return _orig_torch_save(obj, f, *args, **kwargs)

    target = _os.fspath(f)
    tmp = target + '.tmp'

    for attempt in range(3):
        try:
            _orig_torch_save(obj, tmp, *args, **kwargs)
            if _os.path.exists(target):
                _os.remove(target)
            _os.rename(tmp, target)
            return
        except (ValueError, OSError) as e:
            # 清理失败的临时文件
            if _os.path.exists(tmp):
                try:
                    _os.remove(tmp)
                except Exception:
                    pass
            if attempt == 2:
                raise
            print(f"  ⚠ torch.save 失败（{e}），第 {attempt+2}/3 次重试...")
            _time.sleep(2.0)

torch.save = _safe_torch_save

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH    = PROJECT_ROOT / 'pet_recognition' / 'pet_gate.yaml'
# 日志和模型存到 D 盘纯英文路径，避免中文路径 + Windows Defender 导致 torch.save 失败
LOG_DIR      = Path(r'D:/pet_train_logs')
MODEL_OUT    = PROJECT_ROOT / 'pet_recognition' / 'models' / 'pet_gate.pt'

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    # ── 设备选择 ──
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    else:
        device = 'cpu'
        print("⚠ CUDA 不可用，将使用 CPU 训练（会非常慢）")

    # ── 加载预训练模型 ──
    print("加载 YOLOv8m 预训练权重...")
    model = YOLO('yolov8m.pt')

    # ── 训练 ──
    print(f"\n开始训练（device={device}, batch=8, amp=True）...\n")
    model.train(
        data=str(YAML_PATH),
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.001,
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        freeze=10,
        augment=True,
        device=device,
        project=str(LOG_DIR),
        name='yolov8m_pet_gate',
        exist_ok=True,
        amp=True,
        patience=15,
        verbose=True,
        workers=0,          # 禁用多进程加载，避免 WinError 1455（页面文件太小）
    )

    # ── 拷贝最优权重 ──
    best_path = LOG_DIR / 'yolov8m_pet_gate' / 'weights' / 'best.pt'
    if best_path.exists():
        shutil.copy(str(best_path), str(MODEL_OUT))
        print(f"\n✓ 模型已保存到 {MODEL_OUT}")
    else:
        print(f"\n⚠ 未找到 best.pt（{best_path}），请检查训练日志")

    # ── 测试集评估 ──
    print("\n测试集评估...")
    metrics = model.val(data=str(YAML_PATH), split='test', device=device)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics.box, 'top1'):
        print(f"  Top-1:    {metrics.box.top1:.4f}")


if __name__ == '__main__':
    main()
