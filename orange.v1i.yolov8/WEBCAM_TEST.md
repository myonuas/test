# YOLOv8 摄像头检测测试说明 (orange检测版本)

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install ultralytics opencv-python
```

## 使用方法

### 1. 使用预训练模型测试（推荐先试用）

使用YOLOv8n轻量级预训练模型（可以检测80种常见物体）：

```bash
python webcam_test.py
```

### 2. 使用训练好orange检测模型

如果你已经用这个数据集训练好orange检测模型：

```bash
python webcam_test.py --model your_trained_model.pt
```

### 3. 调整置信度阈值

默认置信度为0.5，可以调整：

```bash
q  # 降低阈值，检测更多目标
python webcam_test.py --conf 0.7  # 提高阈值，更严格
```

## 操作指南

- **打开摄像头**：运行脚本后自动打开内置摄像头
- **退出程序**：按 `q` 键
- **保存截图**：按 `s` 键（保存为 detection_result.jpg）

## 训练自己orange模型

如果你想用这个数据集训orange检测模型：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 训练
results = model.train(data="data.yaml", epochs=100, imgsz=640)

# 训练完成后，模型会保存在 runs/detect/train/weights/best.pt

# 然后使用训练好的模型进行摄像头测试
# python webcam_test.py --model runs/detect/train/weights/best.pt
```

## 注意事项

1. 确保摄像头没有被其他程序占用
2. 如果摄像头无法打开，尝试更改摄像头索引（将 `cap = cv2.VideoCapture(0)` 中的0改为1）
3. 预训练模型可以检测常见物体，但不是专门针对chip的
4. 要获得最orange检测效果，需要使用数据集训练专门的模型

pip install ultralytics pyttsx3 opencv-python
python voice_detection.py

## 运行训练脚本

```bash
python train_model.py
```

## 训练完成后运行

```bash
python webcam_test.py --model runs/detect/runs/detect/weights/best.pt
```