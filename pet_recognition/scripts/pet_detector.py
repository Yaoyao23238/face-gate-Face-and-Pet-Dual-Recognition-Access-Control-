# pet_detector.py — YOLOv8m 宠物品种检测推理封装
# 供 detect_fixed.py（Phase 2）import 使用
import json
from pathlib import Path
from ultralytics import YOLO


class PetDetector:
    """
    YOLOv8m 宠物品种检测器（非阻塞，帧驱动）

    Usage:
        detector = PetDetector("pet_recognition/config/pet_config.json")
        result = detector.detect(frame)   # → (中文名, 置信度, 英文名) 或 None
    """

    def __init__(self, config_path):
        config_path = Path(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        self.model_path = cfg['model_path']
        self.confidence_threshold = cfg['confidence_threshold']
        self.allowlist = set(cfg['allowlist'])
        self.input_size = cfg.get('input_size', 640)
        self.en_to_zh = cfg.get('en_to_zh', {})
        self.zh_to_en = cfg.get('zh_to_en', {})

        # 延迟加载模型（避免导入时就把模型加载进内存）
        self._model = None

        # 验证白名单
        for name_zh in self.allowlist:
            if name_zh not in self.zh_to_en:
                print(f"[PetDetector] ⚠ 白名单品种'{name_zh}'无对应英文品种，"
                      f"当前 37 品种中无法匹配到此名称，将永远不触发放行。")

    @property
    def model(self):
        """延迟加载"""
        if self._model is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"模型文件不存在: {self.model_path}\n"
                    f"请先运行 train.py 训练模型"
                )
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame):
        """
        单帧推理

        Args:
            frame: numpy array (H, W, 3), BGR 格式（OpenCV 默认）

        Returns:
            (breed_zh, conf, breed_en, box, granted) 或 None（无检测框时）
            - granted=True  → 置信度达标 + 白名单命中 → 放行
            - granted=False → 未达标或不在白名单 → 拒绝
        """
        results = self.model(frame, imgsz=self.input_size, verbose=False)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        # 取置信度最高的检测
        best_idx = results[0].boxes.conf.argmax()
        cls_id   = int(results[0].boxes.cls[best_idx])
        conf     = float(results[0].boxes.conf[best_idx])

        breed_en = self.model.names[cls_id]
        breed_zh = self.en_to_zh.get(breed_en, breed_en)
        box      = results[0].boxes.xyxy[best_idx].cpu().numpy().astype(int)

        granted = conf >= self.confidence_threshold and breed_zh in self.allowlist
        return (breed_zh, conf, breed_en, tuple(box), granted)

    def detect_raw(self, frame):
        """
        无过滤推理（调试/评估用）

        Returns:
            [(breed_en: str, confidence: float), ...]  或空列表
        """
        results = self.model(frame, imgsz=self.input_size, verbose=False)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        detections = []
        for i in range(len(results[0].boxes)):
            cls_id = int(results[0].boxes.cls[i])
            conf   = float(results[0].boxes.conf[i])
            detections.append((self.model.names[cls_id], conf))
        return sorted(detections, key=lambda x: -x[1])  # 按置信度降序
