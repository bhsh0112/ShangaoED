import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
import signal
import sys
from  judger import Judger
import argparse
from pathlib import Path
import os
import json

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

class EventDetctor:
    def __init__(self, yolov10_model,input_path,output_path):
        if not yolov10_model:
            raise ValueError("YOLOv10 model cannot be None")
        self.model = yolov10_model
        self.tracked_data = []
        self.frame = []
        self.detection_interval = 1  # Interval for stats (seconds)
        self.last_stat_time = time.time()
        self.vehicle_data=[]
        self.interval=0
        self.frame_events = []
        self.video_filename = os.path.basename(input_path)  # 记录视频文件名
        self.fps=0
        self.traffic_jam_threshold = 10  # Vehicle count threshold
        self.speed_threshold = 5  # Speed threshold (km/h)
        self.is_traffic_jam = False
        self.output_path=output_path
        self.input_path=input_path
        self.video_writer = None  # 延后到实际打开输入源后创建
        self.stop_requested = False

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

    def output(self,jam_result,frame_count,detected_objects):
        jam, park, people = jam_result[:3]

        frame_data = {
            "frame": frame_count,
            "timestamp": frame_count / self.fps,  # 计算时间戳
            "event":{
                "jam": bool(jam),
                "parked": bool(park),
                "people": bool(people)
            },
            "objects": detected_objects
            
        }

        if jam:
            # print("success")
            if people:#jam+people
                cv2.putText(self.frame, JAM_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
                cv2.putText(self.frame, PEOPLE_MESSAGE, (30, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
            else:#only jam
                cv2.putText(self.frame, JAM_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
        else:
            if park:
                if people:#park+people
                    cv2.putText(self.frame, PARKED_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
                    cv2.putText(self.frame, PEOPLE_MESSAGE, (30, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
                else:#only park
                    cv2.putText(self.frame, PARKED_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
            else:
                if people:#only people
                    cv2.putText(self.frame, PEOPLE_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,EVENT_COLOR , 2)
                else:#no event
                    cv2.putText(self.frame, NORMAL_MESSAGE, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 1,NORMAL_COLOR , 2)

        # event_data["status"] = status
        self.frame_events.append(frame_data)


    def save_events_to_json(self):
        # 构建与输出视频同名的JSON文件路径
        json_path = os.path.splitext(self.output_path)[0] + "_events.json"
        
        # 构建完整的输出数据结构
        output_data = {
            "video_filename": self.video_filename,
            "fps": float(self.fps),
            "total_frames": len(self.frame_events),
            "result": self.frame_events
        }
        
        # 写入JSON文件
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"事件数据已保存至: {json_path}")
    
    def request_stop(self):
        self.stop_requested = True

    def run_tracking(self, video_path):
        """Run YOLOv10 model for object detection or RTSP stream"""
        print(f"Processing source: {video_path}")

        # 更稳健的 RTSP 打开方式：设置 FFmpeg 选项并尝试使用 CAP_FFMPEG 后端
        cap = None
        is_rtsp = str(video_path).lower().startswith("rtsp://")
        if is_rtsp:
            # 通过环境变量为 FFmpeg 传入选项（OpenCV 4.5+ 支持）
            # stimeout/rw_timeout 单位为微秒，优先使用 TCP，降低掉线概率
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|rw_timeout;10000000|max_delay;500000|buffer_size;10485760"
            # 第一次尝试：默认后端
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # 第二次尝试：强制使用 FFmpeg 后端
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(video_path)

        # 进一步的重试机制（短暂重试几次）
        retry = 0
        while (cap is None or not cap.isOpened()) and retry < 3:
            retry += 1
            print(f"无法打开源，重试 {retry}/3 ...")
            if cap is not None:
                cap.release()
            time.sleep(1.0)
            if is_rtsp:
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(video_path)

        assert cap is not None and cap.isOpened(), "Cannot open video/stream source"

        # 降低缓冲，提升实时性
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        # 获取视频的帧率、宽度和高度（RTSP可能返回0，需兜底）
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0
        self.fps = float(fps)
        self.interval = 1.0 / self.fps if self.fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            # 读一帧以获取尺寸
            ok, probe_frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read first frame to determine size")
            height, width = probe_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 定义视频编码器和输出文件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        frame_count = 0

        window_name = "EventDet"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while cap.isOpened() and not self.stop_requested:
            # for _ in range(2):  # Discard the most recent 2 frames
            ret, self.frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # yolov10 track
            tracks = self.model.track(self.frame, conf=0.1, iou=0.1, persist=True, show=False, verbose=False)
            # Ensure tracks is not None
            if tracks is None :
                print("Error: Tracks data is None.")
                continue  # Skip this frame if no tracks

            detected_objects = []

            # Draw detection results
            jam_result=[False,False,False]# jam, park, people
            judger=Judger(None,None,[False,False,False],[])
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
                    judger.current_data=current_data
                    judger.prev_data=prev_data
                    judger.main()
                    jam_result=judger.result

                    detected_objects.append({
                        "track_id": int(track_id),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": {
                            "x_min": float(x_min),
                            "y_min": float(y_min),
                            "x_max": float(x_max),
                            "y_max": float(y_max)
                        },
                        "center": {
                            "x": float(current_position[0]),
                            "y": float(current_position[1])
                        },
                        "size": {
                            "width": float(x_max - x_min),
                            "height": float(y_max - y_min)
                        },
                        "speed": float(speed)
                    })
                    
                    # jam_vehicle_num=judger.jam_vehicle_num

                    #update self.vehicle_data(merge)
                    self.update_data(track_id,current_data)
                
            #draw the message about parking
            self.output(jam_result,frame_count,detected_objects)
            # Show and output
            cv2.imshow(window_name, self.frame)
            self.video_writer.write(self.frame)

            # 文本输出（简要事件信息）
            try:
                jam, park, people = jam_result[:3]
            except Exception:
                jam, park, people = False, False, False
            ts_sec = frame_count / self.fps if self.fps else 0
            print(f"帧 {frame_count} | 时间 {ts_sec:.2f}s | 事件: 拥堵={bool(jam)} 停车={bool(park)} 行人={bool(people)} | 目标数={len(detected_objects)}")

            frame_count += 1 

            # Press 'q' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

        self.save_events_to_json()

def signal_handler(sig, frame):
    # 优雅停止，确保资源释放并保存视频与事件JSON
    try:
        if 'yolo_tracker' in globals() and yolo_tracker is not None:
            yolo_tracker.request_stop()
            print("\n中断信号已接收，正在保存视频与事件数据...")
        else:
            print("\n中断信号已接收。")
    except Exception as e:
        print(f"\n中断处理异常: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--weights",  type=str, default=ROOT / "yolov10n-shangao-v2.pt", help="model path or triton URL")
parser.add_argument("--source", type=str, default=ROOT / "data/test/test.mp4", help="file/dir/URL/glob/screen/0(webcam)")
parser.add_argument("--output", type=str, default="output/", help="output path")

args = parser.parse_args()

# Start YOLOv10Tracker
input_path = str(args.source)
output_path = str(args.output)
if os.path.isdir(output_path):
    # 针对RTSP/RTMP/HTTP流或摄像头，生成带时间戳的文件名
    if str(input_path).startswith(("rtsp://","rtmp://","http://","https://")) or str(input_path).isdigit():
        input_filename = f"stream_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    else:
        input_filename = os.path.basename(input_path)
    output_path = os.path.join(output_path, input_filename)

print("Final output path:", output_path)
yolov10_model = YOLO(args.weights)
# print(output_path)
yolo_tracker = EventDetctor(yolov10_model,input_path=input_path,output_path=output_path)
# 捕获 Ctrl+C 信号（在创建 tracker 之后安装处理器，方便访问实例）
signal.signal(signal.SIGINT, signal_handler)
yolo_tracker.run_tracking(input_path)
