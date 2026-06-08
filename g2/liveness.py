# liveness.py — 人脸活体检测（帧间微动法）
# 原理：真人面部有自然微抖动，照片/视频回放完全静止
# 零新依赖，仅需 MTCNN 输出的检测框坐标

import time
import numpy as np


class LivenessDetector:
    """
    Usage:
        lv = LivenessDetector(collect_frames=15, threshold=3.0)

        # 人脸识别通过后，每帧传入检测框
        passed, progress = lv.check(box)   # box = (x1,y1,x2,y2)

        # 识别失败或长时间未通过时重置
        lv.reset()
    """

    def __init__(self, collect_frames=15, threshold=3.0):
        self.collect_frames = collect_frames
        self.threshold = threshold
        self.centers = []
        self.start_time = None

    def start(self):
        self.centers = []
        self.start_time = time.time()

    def reset(self):
        self.centers = []
        self.start_time = None

    def check(self, box):
        """
        返回 (passed, progress)
        - passed: bool, 活体检测是否通过
        - progress: float, 0~1 采集进度
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.centers.append((cx, cy))

        if len(self.centers) < self.collect_frames:
            return (False, len(self.centers) / self.collect_frames)

        # 计算框中心在 xy 方向上的帧间标准差
        pts = np.array(self.centers)
        movement = float(np.std(pts[:, 0]) + np.std(pts[:, 1]))
        passed = movement >= self.threshold
        return (passed, 1.0)
