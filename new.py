import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLOv10
import signal
import sys
import queue
import threading

# 初始化YOLOv10模型
yolov10_model = YOLOv10("./fake-yolo.pt")

# 设置测速线段的两个端点,一条直线,(x,y)
line_pts = [(0, 615), (1920, 615)]
# 初始化速度估计器
from ultralytics.solutions import speed_estimation, object_counter
speed_obj = speed_estimation.SpeedEstimator()
counter_obj = object_counter.ObjectCounter()

# 公式参数
params_width = [-30843.21, 300.83]
params_height = [-29908.30, 284.79]

# 根据拟合结果构造函数，输入y值，输出框的宽度和高度
def calculate_box_size(y):
    width = np.polyval(params_width, y)  # 计算宽度
    height = np.polyval(params_height, y)  # 计算高度
    return int(width), int(height)

# 创建 VideoWriter 对象
frame_width = 1920
frame_height = 1080
output_filename = "output.mp4"
video_writer = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),  # 编码器
    30,  # 帧率
    (frame_width, frame_height)  # 视频帧大小
)

class YOLOv10Tracker:
    def __init__(self, yolov10_model):
        if not yolov10_model:
            raise ValueError("YOLOv10 model cannot be None")
        self.model = yolov10_model
        self.tracked_data = []
        self.frame = []
        self.vehicle_data = pd.DataFrame(columns=[
            'id', 'Time', 'type', 'x', 'y', 'length', 'width'
        ])
        self.traffic_jam_threshold = 10  # 车辆数量阈值
        self.is_traffic_jam = False

    def get_frame(self):
        return self.frame

    def generate_statistics(self, frame):
        """生成并显示统计信息"""
        if not self.vehicle_data.empty:
            # 示例统计区域（需要根据实际坐标调整）
            AREA = (0, 1920, 0, 1080)  # x1, x2, y1, y2

            # 统计车辆数量
            vehicle_count = len(self.vehicle_data)

            # 判断是否交通拥堵
            if vehicle_count >= self.traffic_jam_threshold:
                self.is_traffic_jam = True
                # 在画面中央显示“交通拥堵”提示
                cv2.putText(frame, "traffic jam", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                self.is_traffic_jam = False

            # 在画面左上角显示统计信息
            text_y = 50
            cv2.putText(frame, f"vehicle num: {vehicle_count}", (20, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return vehicle_count
        else:
            # 车辆数据为空时返回默认值
            return 0

    def run_tracking(self, video_path):
        """运行YOLOv10模型进行目标检测"""
        print(f"开始处理视频文件：{video_path}")
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "无法打开视频文件"
        
        while cap.isOpened():
            ret, self.frame = cap.read()
            if not ret:
                break

            # 数据收集逻辑
            current_time = time.time()
            self.generate_statistics(self.frame)

            # 执行YOLOv10目标检测
            tracks = self.model.track(self.frame, conf=0.05, iou=0.1, persist=True, show=False, verbose=False)
            
            # 确保 tracks 不为 None
            if tracks is None:
                continue

            # 清空之前的车辆数据
            self.vehicle_data = pd.DataFrame(columns=[
                'id', 'Time', 'type', 'x', 'y', 'length', 'width'
            ])

            # 绘制检测结果
            for result in tracks:
                boxes = result.boxes  # 获取边界框数据
                confidences = boxes.conf  # 获取置信度数据
                class_ids = boxes.cls  # 获取类别索引数据
                names = result.names if hasattr(result, 'names') else {}

                # 遍历每个检测框
                for i, box in enumerate(boxes.xyxy):
                    x_min, y_min, x_max, y_max = box  # 获取框的坐标
                    confidence = confidences[i]  # 获取当前框的置信度
                    class_id = int(class_ids[i])  # 获取类别 ID

                    if confidence < 0.05:
                        continue

                    # 绘制检测框
                    color = (0, 255, 0)  # 绿色框
                    cv2.rectangle(self.frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                    # 绘制类别和置信度
                    class_name = names[class_id] if names and class_id in names else "unknown"
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(self.frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 添加到车辆数据中
                    new_data = pd.DataFrame([{
                        'id': result[-1] if len(result) > 0 else -1,  # 使用track_id
                        'Time': current_time,
                        'type': class_id, 
                        'x': (x_min + x_max) / 2,
                        'y': (y_min + y_max) / 2,
                        'length': 6,  # 示例换算比例
                        'width': 3
                    }])

                    self.vehicle_data = pd.concat([self.vehicle_data, new_data])

            # 显示当前帧
            cv2.imshow("Frame", self.frame)
            video_writer.write(self.frame)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

def signal_handler(sig, frame):
    print("\n录制结束！视频已保存为", output_filename)
    sys.exit(0)

# 捕获 Ctrl+C 信号
signal.signal(signal.SIGINT, signal_handler)

# 启动YOLOv10Tracker
url = "/mnt/sda1/shangao/山高数据2.20/停车/停车02.mp4" 
yolo_tracker = YOLOv10Tracker(yolov10_model)
yolo_tracker.run_tracking(url)
