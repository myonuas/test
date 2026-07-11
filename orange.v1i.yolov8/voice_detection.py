"""
物体检测语音播报脚本
检测到物体后，用语音播报物体的标签名称
"""

import cv2
import time
import os
from ultralytics import YOLO
import pyttsx3
import threading

class VoiceDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        初始化语音检测器
        :param model_path: YOLO模型路径
        :param confidence_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.last_announcement = ""
        self.announcement_lock = threading.Lock()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 语速
        self.engine.setProperty('volume', 0.9)  # 音量

        # 设置语音属性
        voices = self.engine.getProperty('voices')
        # 选择合适的语音（通常voices[0]是男性，voices[1]是女性）
        self.engine.setProperty('voice', voices[1].id)

        # 获取训练时的类别名称（只播报这些类别）
        self.trained_classes = set(self.model.names.values())

        print("语音检测器初始化完成")
        print(f"使用模型: {model_path}")
        print(f"置信度阈值: {confidence_threshold}")
        print(f"训练类别: {self.trained_classes}")

    def announce(self, text):
        """
        语音播报文本
        :param text: 要播报的文本
        """
        with self.announcement_lock:
            # 避免重复播报相同的文本
            if text == self.last_announcement:
                return

            self.last_announcement = text
            print(f"播报: {text}")

            # 在单独的线程中播报，避免阻塞主线程
            def speak():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"语音播报错误: {e}")

            threading.Thread(target=speak, daemon=True).start()

    def detect_and_announce(self, image):
        """
        检测图像并播报结果（只播报训练类别中的物体）
        :param image: 输入图像
        :return: 检测后的图像
        """
        results = self.model(image, conf=self.conf_threshold)

        detected_objects = set()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                confidence = float(box.conf[0])

                if confidence >= self.conf_threshold and label in self.trained_classes:
                    detected_objects.add(label)

        # 如果检测到训练类别中的物体，播报
        if detected_objects:
            # 合并所有检测到的物体名称
            announcement_text = "检测到 " + "、".join(detected_objects)
            self.announce(announcement_text)

        return results[0].plot()

    def detect_from_camera(self, camera_index=0):
        """
        从摄像头进行实时检测和播报
        :param camera_index: 摄像头索引
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("开始实时检测，按 'q' 键退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break

            # 检测并播报
            result_frame = self.detect_and_announce(frame)

            # 显示结果
            cv2.imshow("Object Detection", result_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # 图片检测功能已移除，只保留摄像头检测

def get_trained_model_path():
    """
    获取训练好的模型路径
    :return: 最佳模型路径，如果不存在则返回默认模型
    """
    trained_model_path = "runs/detect/runs/detect/weights/best.pt"
    if os.path.exists(trained_model_path):
        print(f"使用训练好的模型: {trained_model_path}")
        return trained_model_path
    else:
        print("未找到训练好的模型，使用默认模型")
        return "yolov8n.pt"

if __name__ == "__main__":
    # 获取训练好的模型路径
    model_path = get_trained_model_path()

    # 创建语音检测器实例
    detector = VoiceDetector(model_path=model_path, confidence_threshold=0.5)

    # 直接启动摄像头实时检测
    print("\n启动摄像头实时检测...")
    detector.detect_from_camera()