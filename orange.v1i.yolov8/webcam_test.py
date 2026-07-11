"""
YOLOv8摄像头实时检测脚本 (orange检测版本)
使用方法：
1. 使用预训练模型测试（默认）：python webcam_test.py
2. 使用训练好的模型：python webcam_test.py --model your_model.pt
"""

import cv2
import argparse
from ultralytics import YOLO

def run_webcam_detection(model_path='yolov8n.pt', confidence=0.5):
    """
    运行摄像头实时检测

    Args:
        model_path: 模型路径，默认使用YOLOv8n预训练模型
        confidence: 置信度阈值
    """
    print(f"正在加载模型: {model_path}")

    try:
        model = YOLO(model_path)
        print(f"模型加载成功！")

        # 获取摄像头
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        print("摄像头已打开！按 'q' 键退出，按 's' 键保存截图")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            # 进行检测
            results = model(frame, conf=confidence, verbose=False)

            # 绘制结果
            annotated_frame = results[0].plot()

            # 显示结果
            cv2.imshow('YOLOv8 实时检测 - Chip', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存截图
                cv2.imwrite('detection_result.jpg', annotated_frame)
                print("已保存检测结果到 detection_result.jpg")

        cap.release()
        cv2.destroyAllWindows()
        print("程序结束")

    except Exception as e:
        print(f"发生错误: {e}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLOv8摄像头检测 (orange)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型路径（默认使用yolov8n.pt预训练模型）')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值（默认0.5）')

    args = parser.parse_args()
    run_webcam_detection(args.model, args.conf)

if __name__ == "__main__":
    main()