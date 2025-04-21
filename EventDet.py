import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLOv10
import signal
import sys
from  judger import Judger
import argparse
from pathlib import Path
import os

PARKED_MESSAGE="there are parked cars!"
JAM_MESSAGE="jam!"
PEOPLE_MESSAGE="there are people!"
NORMAL_MESSAGE="everything is ok"

EVENT_COLOR=(0,0,255)
NORMAL_COLOR=(0,255,0)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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
    
    def track_match(self,track_id):
        prev_data= dict()
        for data in self.vehicle_data:
            if data['id']==track_id:
                prev_data=data
                break
        return prev_data
    
    def calculate_speed(self,prev_data,current_position):
        prev_position=[]
        if(len(prev_data)>0):
            prev_position = [prev_data['x'], prev_data['y']]

        # Calculate speed if previous position exists
        if len(prev_position) > 0:
            prev_x,prev_y=prev_position
            speed_x = (current_position[0] - prev_x)  # Assuming pixel per frame distance
            speed_y = (current_position[1] - prev_y)  # Assuming pixel per frame distance
            speed = np.sqrt(speed_x ** 2 + speed_y ** 2)
        else:
            speed = 0  # No previous position, so speed is 0
        return speed
    
    def update_data(self,track_id,current_data):
        tmp_flag=True
        for index,data in enumerate(self.vehicle_data):
            if data['id']==track_id:
                self.vehicle_data[index]=current_data
                tmp_flag=False
                break
        if tmp_flag is True:
            self.vehicle_data.append(current_data)

    def output(self,judger):
        if judger.result[0]:#jam
            cv2.putText(self.frame, JAM_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
            if judger.result[2]:#jam+people
                cv2.putText(self.frame, PEOPLE_MESSAGE, (30, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
        else:
            if judger.result[1]:#park
                cv2.putText(self.frame, PARKED_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
                if judger.result[2]:#park+people
                    cv2.putText(self.frame, PEOPLE_MESSAGE, (30, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
            else:
                if judger.result[3]:#people
                    cv2.putText(self.frame, PEOPLE_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
                else:#no event
                    cv2.putText(self.frame, NORMAL_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,NORMAL_COLOR , 2)


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

            # Draw detection results
            result=[False,False,False]#jam,park,people
            for result in tracks:
                if result.boxes.id is None:
                    continue

                boxes = result.boxes  # Get bounding box data
                confidences = boxes.conf  # Get confidence scores
                class_ids = boxes.cls  # Get class IDs
                names = result.names if hasattr(result, 'names') else {}
                track_ids=result.boxes.id.int().cpu().tolist()

                jam_vehicle_num=0
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

                    # match previous data
                    prev_data=self.track_match(track_id)

                    #get previous posituon
                    speed=self.calculate_speed(prev_data=prev_data,current_position=current_position)

                    # get current data
                    current_data = {
                        'id': track_id,  # Use track_id
                        'class':class_name,
                        'Time': current_time,
                        'type': class_id, 
                        'x': current_position[0],
                        'y': current_position[1],
                        'size_w':x_max-x_min,
                        'size_h':y_max-y_min,
                        'speed': speed, 
                        'yaw': 0,
                        'length': 6,  # Example conversion ratio
                        'width': 3,
                        
                    }
                    judger=Judger(current_data,prev_data,result,jam_vehicle_num)
                    ED_message,ED_color,jam_vehicle_num=judger.main()
                    result=judger.result

                    #update self.vehicle_data(merge)
                    self.update_data(track_id,current_data)
                
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
    print("\nRecording stopped! Video saved as", output_path)
    sys.exit(0)

# Capture Ctrl+C signal
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "weights/fake-yolo.pt", help="model path or triton URL")
parser.add_argument("--source", type=str, default=ROOT / "data/test.mp4", help="file/dir/URL/glob/screen/0(webcam)")
parser.add_argument("--output", type=str, default=ROOT / "output/", help="output path")

args = parser.parse_args()

# Start YOLOv10Tracker
input_path = str(args.source)
output_path = str(args.output)
yolov10_model = YOLOv10(args.weights)
yolo_tracker = YOLOv10Tracker(yolov10_model,input_path=input_path,output_path=output_path)
yolo_tracker.run_tracking(input_path)
