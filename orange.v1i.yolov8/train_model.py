"""
使用orange数据集训练YOLOv8模型
训练完成后，可以使用 webcam_test.py --model best.pt 进行摄像头测试
"""

from ultralytics import YOLO
import os

def train_orange_model():
    print("=" * 50)
    print("开始训练orange检测模型")
    print("=" * 50)

    # 检查数据集是否存在
    if not os.path.exists("data.yaml"):
        print("错误：找不到 data.yaml 文件")
        return

    # 选择模型大小
    print("\n请选择模型大小：")
    print("1. yolov8n (nano) - 最快，精度较低")
    print("2. yolov8s (small) - 平衡")
    print("3. yolov8m (medium) - 精度更高")
    print("4. yolov8l (large) - 精度最高，较慢")

    choice = input("\n请输入选择 (1-4，默认1): ").strip() or "1"

    model_map = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt",
        "3": "yolov8m.pt",
        "4": "yolov8l.pt"
    }

    model_name = model_map.get(choice, "yolov8n.pt")
    print(f"\n选择模型: {model_name}")

    # 训练参数
    epochs = int(input("\n训练轮数 (默认50): ").strip() or "50")
    imgsz = int(input("图像尺寸 (默认640): ").strip() or "640")
    batch = int(input("批次大小 (默认16): ").strip() or "16")

    print(f"\n训练参数:")
    print(f"  - 模型: {model_name}")
    print(f"  - 轮数: {epochs}")
    print(f"  - 图像尺寸: {imgsz}")
    print(f"  - 批次大小: {batch}")
    print(f"  - 数据集: data.yaml")
    print("\n开始训练...")

    try:
        # 加载模型
        model = YOLO(model_name)

        # 训练模型
        results = model.train(
            data="data.yaml",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device="cpu",  # 如果有GPU可以改为 "0"
            project="runs",
            name="detect",
            exist_ok=True
        )

        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)
        print(f"\n最佳模型已保存到: runs/detect/train/weights/best.pt")
        print(f"最后模型已保存到: runs/detect/train/weights/last.pt")
        print("\n使用方法:")
        print(f"python webcam_test.py --model runs/detect/train/weights/best.pt")

    except Exception as e:
        print(f"\n训练失败: {e}")
        print("\n请检查:")
        print("1. 是否安装了 ultralytics: pip install ultralytics")
        print("2. 数据集是否完整")
        print("3. 内存是否充足")

if __name__ == "__main__":
    train_orange_model()