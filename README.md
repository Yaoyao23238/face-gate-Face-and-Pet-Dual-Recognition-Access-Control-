# Face Gate — 人脸+宠物双模智能门禁系统

基于深度学习的人脸识别与宠物品种检测双模门禁系统。支持实时摄像头推理、活体检测、Web 可视化管理。

## 功能特性

- **人脸识别**：MTCNN + FaceNet (InceptionResnetV1)，余弦相似度匹配
- **活体检测**：帧间微动法，防照片/视频攻击（仅人脸分支）
- **宠物识别**：YOLOv8m 微调，Oxford-IIIT Pet 37 品种，mAP50 > 93%
- **Web 界面**：Flask 深色仪表盘，实时画面 + 管理面板
- **鲁棒性**：14 种扰动测试，评分 98.3/100

## 项目结构

```
face_gate/
├── g2/                          # 人脸识别核心
│   ├── train.py                 # 人脸特征训练
│   ├── detect_fixed.py          # 门禁主程序（桌面版）
│   ├── evaluate_accuracy.py     # 准确率评测
│   ├── evaluate_robustness.py   # 鲁棒性测试
│   ├── liveness.py              # 活体检测模块
│   └── animation.py             # Tkinter 门禁动画
├── pet_recognition/             # 宠物识别
│   ├── scripts/
│   │   ├── train.py             # YOLOv8m 训练
│   │   ├── download_dataset.py  # 数据集下载
│   │   └── pet_detector.py      # 推理封装
│   ├── config/
│   │   └── pet_config.example.json
│   └── pet_gate.yaml            # 数据集配置
├── templates/
│   └── index.html               # Web 前端
├── web_gate.py                  # Flask Web 后端
├── config.example.json          # 人脸识别配置模板
└── requirements.txt
```

## 快速开始

### 1. 环境要求

- Python 3.9+
- PyTorch 2.0+ (CUDA 推荐)
- NVIDIA GPU (RTX 3060+ 推荐，6GB VRAM 最低)
- Windows / Linux

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 人脸注册

1. 将注册人脸照片放入 `knowns/<姓名>/` 目录（每人 5-20 张）
2. 运行训练：
```bash
python g2/train.py
```
3. 复制配置：
```bash
cp config.example.json config.json
# 编辑 config.json，填入训练出的 identities 和 threshold
```

### 4. 宠物训练

```bash
# 下载 Oxford-IIIT Pet 数据集
python pet_recognition/scripts/download_dataset.py

# 训练 YOLOv8m（约 1.5 小时, GPU 推荐）
python pet_recognition/scripts/train.py

# 配置白名单
cp pet_recognition/config/pet_config.example.json pet_recognition/config/pet_config.json
# 编辑 allowlist 字段，填入需要放行的品种中文名
```

### 5. 启动门禁

**桌面版**：
```bash
python g2/detect_fixed.py
```

**Web 版**：
```bash
python web_gate.py
# 浏览器打开 http://localhost:5000
```

### 6. 评测

```bash
# 人脸识别准确率
python g2/evaluate_accuracy.py

# 鲁棒性测试
python g2/evaluate_robustness.py
```

## 注意事项

- `knowns/`、`test/`、`identity_anchors.npz`、`config.json` 含个人隐私，已加入 `.gitignore`
- 宠物数据集和模型权重需自行下载/训练，不包含在仓库中
- Windows 用户如遇 `WinError 1455`，训练脚本已设 `workers=0`
- 活体检测仅适用于人脸分支，不影响宠物识别

## 许可证

MIT License
