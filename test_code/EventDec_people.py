import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLOv10
import signal
import sys

MIN_SPEED_WEIGHT=1

# Initialize YOLOv10 model
yolov10_model = YOLOv10("weights/fake-yolo.pt")

class YOLOv10Tracker:
    def __init__(self, yolov10_model,input_path,output_path):
        if not yolov10_model:
            raise ValueError("YOLOv10 model cannot be None")
        self.model = yolov10_model
        self.tracked_data = []
        self.frame = []
        self.detection_interval = 1  # Interval for stats (seconds)
        self.last_stat_time = time.time()
        self.vehicle_data=[]
        # self.vehicle_data = pd.DataFrame(columns=[
        #     'id', 'Time', 'type', 'x', 'y', 'xv', 'yv', 
        #     'yaw', 'length', 'width', 'prev_x', 'prev_y'
        # ])  # Added prev_x and prev_y to store previous positions
        self.traffic_jam_threshold = 10  # Vehicle count threshold
        self.speed_threshold = 5  # Speed threshold (km/h)
        self.is_traffic_jam = False

        cap = cv2.VideoCapture(input_path)
        # 获取视频的帧率、宽度和高度
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定义视频编码器和输出文件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def get_frame(self):
        return self.frame

    def run_tracking(self, video_path):
        """Run YOLOv10 model for object detection"""
        print(f"Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Cannot open video file"

        while cap.isOpened():
            # for _ in range(2):  # Discard the most recent 2 frames
            ret, self.frame = cap.read()
            if not ret:
                break
            # if not ret:
            #     break

            current_time = time.time()

            # yolov10 track
            tracks = self.model.track(self.frame, conf=0.1, iou=0.1, persist=True, show=False, verbose=False)
            # Ensure tracks is not None
            if tracks is None :
                print("Error: Tracks data is None.")
                continue  # Skip this frame if no tracks

            parking_message="no parked car"
            parking_color=(0,255,0)
            # Draw detection results
            for result in tracks:
                if result.boxes.id is None:
                    continue

                boxes = result.boxes  # Get bounding box data
                confidences = boxes.conf  # Get confidence scores
                class_ids = boxes.cls  # Get class IDs
                names = result.names if hasattr(result, 'names') else {}
                track_ids=result.boxes.id.int().cpu().tolist()

                # Iterate over each detection box
                for i, box in enumerate(boxes.xyxy):
                    speed=0
                    x_min, y_min, x_max, y_max = box  # Get coordinates
                    confidence = confidences[i]  # Get current box confidence
                    class_id = int(class_ids[i])  # Get class ID
                    track_id=track_ids[i]

                    if confidence < 0.1:
                        continue

                    # Draw bounding box
                    color = (0, 255, 0)  # Green box
                    cv2.rectangle(self.frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                    # Label the box with class and confidence
                    class_name = names[class_id] if names and class_id in names else "unknown"
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(self.frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Get the current position of the vehicle
                    current_position = (x_min + x_max) / 2, (y_min + y_max) / 2
                    # print(class_name)
                    if class_name=="person":
                        # print("11111111111111111111111")
                        cv2.putText(self.frame,"there is a people!", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255) , 2)

            #draw the message about parking
            
            # Show and output
            cv2.imshow("Frame", self.frame)
            self.video_writer.write(self.frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

def signal_handler(sig, frame):
    print("\nRecording stopped! Video saved as", output_filename)
    sys.exit(0)

def isParking(current_data,prev_data):
    min_speed=current_data['size_w']*MIN_SPEED_WEIGHT
    # print(current_data['speed'])
    if len(prev_data)==0:
        return False
    else:
        return current_data['speed']<=0.5 and prev_data['speed']<=0.5

# Capture Ctrl+C signal
signal.signal(signal.SIGINT, signal_handler)

# Start YOLOv10Tracker
input_path = '/home/buaa/sh/ShangaoED/test_people.mp4'
output_path = "output.mp4"
yolo_tracker = YOLOv10Tracker(yolov10_model,input_path=input_path,output_path=output_path)
yolo_tracker.run_tracking(input_path)
