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
from tqdm import tqdm
import glob

PARKED_MESSAGE="there are parked cars!"
JAM_MESSAGE="jam!"
PEOPLE_MESSAGE="there are people!"
BREAKDOWN_MESSAGE="breakdown!"
NORMAL_MESSAGE="everything is ok"

EVENT_COLOR=(0,0,255)
NORMAL_COLOR=(0,255,0)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class EventDetctor:
    def __init__(self, yolov10_model,input_path,output_path, show_window=False, window_name="Event Detection"):
        """
        初始化事件检测器
        
        @param {YOLO} yolov10_model - YOLO 模型实例
        @param {str} input_path - 输入视频路径
        @param {str} output_path - 输出视频路径
        @param {bool} show_window - 是否显示实时可视化窗口
        @param {str} window_name - 窗口名称
        """
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

        # 实时可视化配置
        self.show_window = show_window
        self.window_name = window_name
        self.paused = False  # 暂停状态
        self.display_fps = 0.0  # 显示帧率
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

        # 事件稳定判定与筛选配置（秒级，帧率获取后换算为帧）
        # 所有事件都需要持续1秒后才视为发生
        self.event_on_seconds = 1.0  # 连续满足至少此秒数后判为事件发生
        self.event_off_seconds = 1.0 # 连续不满足至少此秒数后解除事件
        
        # 拥堵事件状态跟踪
        self.jam_confirm_frames = 0
        self.jam_clear_frames = 0
        self.jam_state = False
        self._jam_consecutive = 0
        self._nojam_consecutive = 0
        
        # 停车事件状态跟踪
        self.park_confirm_frames = 0
        self.park_clear_frames = 0
        self.park_state = False
        self._park_consecutive = 0
        self._nopark_consecutive = 0
        
        # 行人事件状态跟踪
        self.people_confirm_frames = 0
        self.people_clear_frames = 0
        self.people_state = False
        self._people_consecutive = 0
        self._nopeople_consecutive = 0
        
        # 故障事件状态跟踪
        self.breakdown_confirm_frames = 0
        self.breakdown_clear_frames = 0
        self.breakdown_state = False
        self._breakdown_consecutive = 0
        self._nobreakdown_consecutive = 0

        # 方向与 ROI 配置
        self.jam_axis = 'x'  # 可选 'x' 或 'y'
        self.roi = None      # 例如 (x1, y1, x2, y2)，默认 None 表示全局

    def get_frame(self):
        return self.frame
    
    def update_display_fps(self):
        """更新显示帧率（用于实时可视化）"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # 每秒更新一次
            self.display_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time
    
    def draw_info_panel(self, frame_count, detected_objects, jam_result):
        """
        在帧上绘制信息面板
        
        @param {int} frame_count - 当前帧数
        @param {list} detected_objects - 检测到的对象列表
        @param {list} jam_result - 事件结果 [jam, park, people, breakdown]
        """
        # 绘制半透明背景面板（调整高度以适应新增的故障状态）
        overlay = self.frame.copy()
        panel_height = 140  # 增加高度以容纳故障状态显示
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, self.frame, 0.4, 0, self.frame)
        
        # 绘制文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        y_offset = 15
        
        # 帧数和时间信息
        ts_sec = frame_count / self.fps if self.fps > 0 else 0
        info_texts = [
            f"Frame: {frame_count} | Time: {ts_sec:.2f}s",
            f"FPS: {self.display_fps:.1f} | Objects: {len(detected_objects)}",
            f"Video: {self.video_filename[:40]}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(self.frame, text, (10, y_offset + i * line_height), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # 事件状态指示器
        jam, park, people, breakdown = jam_result[:4] if len(jam_result) >= 4 else (jam_result[0], jam_result[1], jam_result[2], False)
        status_y = y_offset + 3 * line_height + 5
        
        # 拥堵状态
        jam_color = (0, 0, 255) if jam else (0, 255, 0)
        jam_text = f"拥堵: {'是' if jam else '否'}"
        cv2.putText(self.frame, jam_text, (10, status_y), 
                   font, font_scale, jam_color, thickness)
        
        # 停车状态
        park_color = (0, 0, 255) if park else (0, 255, 0)
        park_text = f"停车: {'是' if park else '否'}"
        cv2.putText(self.frame, park_text, (120, status_y), 
                   font, font_scale, park_color, thickness)
        
        # 行人状态
        people_color = (0, 0, 255) if people else (0, 255, 0)
        people_text = f"行人: {'是' if people else '否'}"
        cv2.putText(self.frame, people_text, (220, status_y), 
                   font, font_scale, people_color, thickness)
        
        # 故障状态（显示在第二行，与其他状态对齐）
        breakdown_color = (0, 0, 255) if breakdown else (0, 255, 0)
        breakdown_text = f"故障: {'是' if breakdown else '否'}"
        cv2.putText(self.frame, breakdown_text, (10, status_y + line_height), 
                   font, font_scale, breakdown_color, thickness)
        
        # 暂停提示
        if self.paused:
            pause_text = "PAUSED - Press SPACE to resume, Q to quit"
            text_size = cv2.getTextSize(pause_text, font, 0.6, 2)[0]
            text_x = (self.frame.shape[1] - text_size[0]) // 2
            text_y = self.frame.shape[0] - 30
            cv2.putText(self.frame, pause_text, (text_x, text_y), 
                       font, 0.6, (0, 255, 255), 2)
    
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
        # 归一到像素/秒
        if self.fps and self.fps > 0:
            speed = speed * float(self.fps)
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
        """
        输出检测结果并绘制可视化信息
        
        @param {list} jam_result - 事件结果 [jam, park, people, breakdown]
        @param {int} frame_count - 当前帧数
        @param {list} detected_objects - 检测到的对象列表
        """
        jam, park, people, breakdown = jam_result[:4] if len(jam_result) >= 4 else (jam_result[0], jam_result[1], jam_result[2], False)

        frame_data = {
            "frame": frame_count,
            "timestamp": frame_count / self.fps,  # 计算时间戳
            "event":{
                "jam": bool(jam),
                "parked": bool(park),
                "people": bool(people),
                "breakdown": bool(breakdown)
            },
            "objects": detected_objects
            
        }

        # 绘制事件消息（优化布局，确保清晰可见）
        # 信息面板高度为140，事件消息从160开始，留出足够间距避免重叠
        message_start_y = 160
        message_offset = 40  # 消息之间的垂直间距
        message_font = cv2.FONT_HERSHEY_SIMPLEX
        message_font_scale = 1.0
        message_thickness = 2
        
        # 收集所有需要显示的事件消息（按优先级排序）
        event_messages = []
        if jam:
            event_messages.append((JAM_MESSAGE, EVENT_COLOR))
        if breakdown:
            event_messages.append((BREAKDOWN_MESSAGE, EVENT_COLOR))
        if park and not jam:  # 拥堵时不显示停车
            event_messages.append((PARKED_MESSAGE, EVENT_COLOR))
        if people:
            event_messages.append((PEOPLE_MESSAGE, EVENT_COLOR))
        
        # 如果没有事件，显示正常状态
        if len(event_messages) == 0:
            event_messages.append((NORMAL_MESSAGE, NORMAL_COLOR))
        
        # 绘制事件消息背景（半透明黑色背景，提高可读性）
        if len(event_messages) > 0:
            # 计算背景框大小
            max_text_width = 0
            total_height = len(event_messages) * message_offset
            for msg, _ in event_messages:
                (text_width, text_height), _ = cv2.getTextSize(msg, message_font, message_font_scale, message_thickness)
                max_text_width = max(max_text_width, text_width)
            
            # 绘制半透明背景（背景框从消息上方10像素开始）
            bg_x1, bg_y1 = 10, message_start_y - 25
            bg_x2, bg_y2 = bg_x1 + max_text_width + 20, bg_y1 + total_height + 10
            overlay = self.frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, self.frame, 0.3, 0, self.frame)
            
            # 绘制所有事件消息
            current_y = message_start_y
            for msg, color in event_messages:
                cv2.putText(self.frame, msg, (15, current_y), 
                           message_font, message_font_scale, color, message_thickness)
                current_y += message_offset

        # 绘制信息面板（如果启用可视化）
        if self.show_window:
            self.draw_info_panel(frame_count, detected_objects, jam_result)

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
        """请求停止处理（用于信号处理）"""
        self.stop_requested = True
    
    def run_tracking(self, video_path):
        """Run YOLOv10 model for object detection or RTSP stream"""
        print(f"Processing source: {video_path}")

        # 更稳健的 RTSP 打开方式：设置 FFmpeg 选项并尝试使用 CAP_FFMPEG 后端
        cap = None
        is_rtsp = str(video_path).lower().startswith("rtsp://")
        is_stream = is_rtsp or str(video_path).lower().startswith(("rtmp://", "http://", "https://")) or str(video_path).isdigit()
        
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
        
        # 基于 fps 计算稳定判定所需帧数（所有事件统一使用1秒）
        self.jam_confirm_frames = max(1, int(round(self.fps * self.event_on_seconds)))
        self.jam_clear_frames = max(1, int(round(self.fps * self.event_off_seconds)))
        self.park_confirm_frames = max(1, int(round(self.fps * self.event_on_seconds)))
        self.park_clear_frames = max(1, int(round(self.fps * self.event_off_seconds)))
        self.people_confirm_frames = max(1, int(round(self.fps * self.event_on_seconds)))
        self.people_clear_frames = max(1, int(round(self.fps * self.event_off_seconds)))
        self.breakdown_confirm_frames = max(1, int(round(self.fps * self.event_on_seconds)))
        self.breakdown_clear_frames = max(1, int(round(self.fps * self.event_off_seconds)))
        
        # 重置所有事件状态（开始处理新视频时）
        self.jam_state = False
        self.park_state = False
        self.people_state = False
        self.breakdown_state = False
        self._jam_consecutive = 0
        self._nojam_consecutive = 0
        self._park_consecutive = 0
        self._nopark_consecutive = 0
        self._people_consecutive = 0
        self._nopeople_consecutive = 0
        self._breakdown_consecutive = 0
        self._nobreakdown_consecutive = 0
        
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
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"创建输出目录: {output_dir}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        
        # 验证 VideoWriter 是否成功初始化
        if not self.video_writer.isOpened():
            raise RuntimeError(f"无法创建输出视频文件: {self.output_path}\n"
                             f"请检查：1) 输出目录是否存在且可写 2) 文件路径是否正确 3) 磁盘空间是否足够")
        
        print(f"视频输出文件已创建: {self.output_path}")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
        has_total = total_frames > 0 and not is_stream
        
        # 始终显示进度条
        pbar = tqdm(total=total_frames if has_total else None, desc=f"Processing {os.path.basename(video_path) if not is_stream else 'stream'}", unit="frame")
        
        # 如果启用可视化，显示提示信息
        if self.show_window:
            print(f"实时可视化已启用 - 窗口名称: {self.window_name}")
            print("控制说明: Q-退出, SPACE-暂停/继续, ESC-退出")

        # 初始化显示FPS计算
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

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
            # 安全初始化 jam_result（用于累积所有车辆的事件状态）
            jam_result=[False,False,False,False] # [jam, park, people, breakdown]
            judger=Judger(None,None,[False,False,False,False],[])
            # 配置 Judger 的方向与 ROI
            judger.jam_axis = self.jam_axis
            judger.roi = self.roi
            # 配置图像高度（用于基于距离的速度阈值计算）
            judger.image_height = height
            # 重置拥堵车辆信息列表（每帧开始时清空）
            judger.jam_vehicle_info = []
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

                    if confidence < 0.3:
                        continue

                    # Get the current position of the vehicle
                    current_position = (x_min + x_max) / 2, (y_min + y_max) / 2

                    # match previous data
                    prev_data=self.track_match(track_id)

                    #get previous posituon
                    speed=self.calculate_speed(prev_data=prev_data,current_position=current_position)

                    # Label the box with class and confidence
                    class_name = names[class_id] if names and class_id in names else "unknown"
                    
                    # get current data
                    current_data = {
                        'id': track_id,  # Use track_id
                        'class':class_name,
                        'Time': current_time,
                        'frame': frame_count,
                        'fps': self.fps,
                        'type': class_id, 
                        'x': current_position[0],
                        'y': current_position[1],
                        'size_w':x_max-x_min,
                        'size_h':y_max-y_min,
                        'speed': speed, 
                        'yaw': 0,
                        'length': 6,  # Example conversion ratio
                        'width': 3,
                        'confidence': float(confidence),
                        
                    }
                    judger.current_data=current_data
                    judger.prev_data=prev_data
                    # 为当前车辆创建独立的结果列表，避免被后续车辆覆盖
                    vehicle_result = [False, False, False, False]
                    judger.result = vehicle_result
                    judger.main()
                    
                    # 获取当前车辆的停车状态（用于绘制橙色框）
                    is_parking = vehicle_result[1] if len(vehicle_result) > 1 else False
                    
                    # 累积所有车辆的事件状态（使用 OR 逻辑：只要有任何一个车辆满足条件，就认为有事件）
                    # 注意：拥堵判断需要在所有车辆处理完后统一进行，所以这里不累积 jam_result[0]
                    if len(vehicle_result) > 1:
                        jam_result[1] = jam_result[1] or vehicle_result[1]  # 停车：累积
                    if len(vehicle_result) > 2:
                        jam_result[2] = jam_result[2] or vehicle_result[2]  # 行人：累积
                    if len(vehicle_result) > 3:
                        jam_result[3] = jam_result[3] or vehicle_result[3]  # 故障：累积
                    if is_parking:
                        # 停车车辆：使用橙色并加粗
                        color = (0, 165, 255)  # 橙色 (BGR格式)
                        thickness = 4  # 加粗线宽
                    else:
                        # 正常车辆：使用绿色
                        color = (0, 255, 0)  # Green box
                        thickness = 2  # 正常线宽
                    
                    # Draw bounding box
                    cv2.rectangle(self.frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

                    # Label the box with class and confidence
                    if class_name=="person":
                        if confidence>=0.8:
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(self.frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            continue
                    else:
                        # 对于车辆，无论是否停车都显示速度和停车阈值
                        speed_threshold = judger.min_speed
                        label1 = f"{class_name}: {confidence:.2f}"
                        label2 = f"Speed: {speed:.2f} | Threshold: {speed_threshold:.2f}"
                        # 绘制第一行标签（类别和置信度）
                        cv2.putText(self.frame, label1, (int(x_min), int(y_min) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        # 绘制第二行标签（速度和阈值）
                        cv2.putText(self.frame, label2, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
            
            # 处理完当前帧所有车辆后，检查拥堵
            # 注意：拥堵判断需要基于当前帧所有慢速车辆，所以要在循环结束后调用
            # 保存已累积的事件状态（停车、行人、故障）
            saved_park = jam_result[1]
            saved_people = jam_result[2] if len(jam_result) > 2 else False
            saved_breakdown = jam_result[3] if len(jam_result) > 3 else False
            
            if not jam_result[0]:  # 如果还没有判定为拥堵
                judger.checkJam()
                # 只更新拥堵状态，保留已累积的其他事件状态
                jam_result[0] = judger.result[0]
                jam_result[1] = saved_park  # 恢复累积的停车状态
                if len(jam_result) > 2:
                    jam_result[2] = saved_people  # 恢复累积的行人状态
                if len(jam_result) > 3:
                    jam_result[3] = saved_breakdown  # 恢复累积的故障状态
            
            # 拥堵状态稳定判定（时间门限与滞回）
            if jam_result[0]:
                self._jam_consecutive += 1
                self._nojam_consecutive = 0
                if not self.jam_state and self._jam_consecutive >= self.jam_confirm_frames:
                    self.jam_state = True
            else:
                self._nojam_consecutive += 1
                self._jam_consecutive = 0
                if self.jam_state and self._nojam_consecutive >= self.jam_clear_frames:
                    self.jam_state = False

            # 停车状态稳定判定（时间门限与滞回）
            if jam_result[1]:
                self._park_consecutive += 1
                self._nopark_consecutive = 0
                if not self.park_state and self._park_consecutive >= self.park_confirm_frames:
                    self.park_state = True
            else:
                self._nopark_consecutive += 1
                self._park_consecutive = 0
                if self.park_state and self._nopark_consecutive >= self.park_clear_frames:
                    self.park_state = False

            # 行人状态稳定判定（时间门限与滞回）
            if len(jam_result) > 2 and jam_result[2]:
                self._people_consecutive += 1
                self._nopeople_consecutive = 0
                if not self.people_state and self._people_consecutive >= self.people_confirm_frames:
                    self.people_state = True
            else:
                self._nopeople_consecutive += 1
                self._people_consecutive = 0
                if self.people_state and self._nopeople_consecutive >= self.people_clear_frames:
                    self.people_state = False

            # 故障状态稳定判定（时间门限与滞回）
            if len(jam_result) > 3 and jam_result[3]:
                self._breakdown_consecutive += 1
                self._nobreakdown_consecutive = 0
                if not self.breakdown_state and self._breakdown_consecutive >= self.breakdown_confirm_frames:
                    self.breakdown_state = True
            else:
                self._nobreakdown_consecutive += 1
                self._breakdown_consecutive = 0
                if self.breakdown_state and self._nobreakdown_consecutive >= self.breakdown_clear_frames:
                    self.breakdown_state = False

            # 使用稳定后的状态进行输出（所有事件都需要持续1秒后才视为发生）
            # 确保jam_result有4个元素
            if len(jam_result) < 4:
                jam_result.extend([False] * (4 - len(jam_result)))
            jam_result[0] = self.jam_state
            jam_result[1] = self.park_state
            jam_result[2] = self.people_state
            jam_result[3] = self.breakdown_state

            #draw the message about parking
            self.output(jam_result,frame_count,detected_objects)
            
            # 更新显示FPS
            if self.show_window:
                self.update_display_fps()
            
            # 写入视频文件（包含可视化内容）
            if self.video_writer is not None and self.video_writer.isOpened():
                self.video_writer.write(self.frame)
            else:
                print(f"警告: VideoWriter 未正确初始化，跳过帧写入")

            # 实时可视化窗口显示
            if self.show_window:
                cv2.imshow(self.window_name, self.frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # Q 或 ESC 退出
                    print("\n用户按下退出键，正在保存...")
                    self.stop_requested = True
                    break
                elif key == ord(' ') or key == ord('p') or key == ord('P'):  # 空格或P暂停/继续
                    self.paused = not self.paused
                    if self.paused:
                        print("已暂停 - 按空格继续")
                    else:
                        print("继续播放")
                
                # 暂停状态处理
                if self.paused:
                    while self.paused and not self.stop_requested:
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q') or key == ord('Q') or key == 27:
                            self.stop_requested = True
                            break
                        elif key == ord(' ') or key == ord('p') or key == ord('P'):
                            self.paused = False
                            print("继续播放")
                            break
                        # 在暂停时也更新显示
                        cv2.imshow(self.window_name, self.frame)
            
            # 文本输出（简要事件信息，仅在非可视化模式或每N帧输出一次）
            if not self.show_window or frame_count % 30 == 0:  # 可视化模式下每30帧输出一次
                try:
                    jam, park, people, breakdown = jam_result[:4] if len(jam_result) >= 4 else (jam_result[0], jam_result[1], jam_result[2], False)
                except Exception:
                    jam, park, people, breakdown = False, False, False, False
                ts_sec = frame_count / self.fps if self.fps else 0
                print(f"帧 {frame_count} | 时间 {ts_sec:.2f}s | 事件: 拥堵={bool(jam)} 停车={bool(park)} 行人={bool(people)} 故障={bool(breakdown)} | 目标数={len(detected_objects)}")

            frame_count += 1 
            if pbar is not None:
                pbar.update(1)

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        if pbar is not None:
            pbar.close()
        
        # 关闭可视化窗口
        if self.show_window:
            cv2.destroyWindow(self.window_name)
            print("可视化窗口已关闭")

        self.save_events_to_json()

def signal_handler(sig, frame):
    """优雅停止，确保资源释放并保存视频与事件JSON"""
    try:
        if 'current_tracker' in globals() and current_tracker is not None:
            current_tracker.request_stop()
            print("\n中断信号已接收，正在保存视频与事件数据...")
        else:
            print("\n中断信号已接收。")
    except Exception as e:
        print(f"\n中断处理异常: {e}")

def get_mp4_files(directory):
    """
    获取目录下所有 MP4 文件（递归搜索）
    
    @param {str} directory - 目录路径
    @returns {list} MP4 文件路径列表
    """
    mp4_files = []
    # 使用 glob 递归搜索所有 .mp4 文件（不区分大小写）
    patterns = [
        os.path.join(directory, "**", "*.mp4"),
        os.path.join(directory, "**", "*.MP4"),
        os.path.join(directory, "*.mp4"),
        os.path.join(directory, "*.MP4")
    ]
    
    for pattern in patterns:
        mp4_files.extend(glob.glob(pattern, recursive=True))
    
    # 去重并排序
    mp4_files = sorted(list(set(mp4_files)))
    return mp4_files

def check_video_has_events(json_path):
    """
    检查视频的 JSON 文件中是否有检测到事件
    
    @param {str} json_path - JSON 文件路径
    @returns {bool} 是否有事件（jam、parked、people 任一为 True）
    """
    try:
        if not os.path.exists(json_path):
            return False
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查 result 中是否有任何帧检测到事件
        if 'result' in data and isinstance(data['result'], list):
            for frame_data in data['result']:
                if 'event' in frame_data:
                    event = frame_data['event']
                    if event.get('jam', False) or event.get('parked', False) or event.get('people', False) or event.get('breakdown', False):
                        return True
        
        return False
    except Exception as e:
        print(f"警告: 读取 JSON 文件失败 {json_path}: {e}")
        return False

def get_videos_with_events(output_dir, video_files):
    """
    获取检测到事件的视频文件名列表
    
    @param {str} output_dir - 输出目录路径
    @param {list} video_files - 输入视频文件路径列表
    @returns {list} 检测到事件的视频文件名列表
    """
    videos_with_events = []
    
    for video_file in video_files:
        # 获取视频文件名（不含路径）
        video_filename = os.path.basename(video_file)
        # 构建对应的 JSON 文件路径
        json_filename = os.path.splitext(video_filename)[0] + "_events.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # 检查是否有事件
        if check_video_has_events(json_path):
            videos_with_events.append(video_filename)
    
    return videos_with_events

def process_single_video(yolov10_model, input_video_path, output_base_path, show_window=False):
    """
    处理单个视频文件
    
    @param {YOLO} yolov10_model - YOLO 模型实例
    @param {str} input_video_path - 输入视频路径
    @param {str} output_base_path - 输出基础路径（目录或文件路径）
    @param {bool} show_window - 是否显示实时可视化窗口
    @returns {bool} 处理是否成功
    """
    global current_tracker
    
    try:
        # 确定输出路径
        if os.path.isdir(output_base_path):
            # 如果输出是目录，使用输入文件名
            input_filename = os.path.basename(input_video_path)
            output_video_path = os.path.join(output_base_path, input_filename)
        else:
            # 如果输出是文件路径，直接使用
            output_video_path = output_base_path
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"处理视频: {input_video_path}")
        print(f"输出路径: {output_video_path}")
        if show_window:
            print(f"实时可视化: 已启用")
        print(f"{'='*60}\n")
        
        # 创建窗口名称（使用视频文件名）
        window_name = f"Event Detection - {os.path.basename(input_video_path)}"
        
        # 创建 EventDetctor 实例
        current_tracker = EventDetctor(
            yolov10_model,
            input_path=input_video_path,
            output_path=output_video_path,
            show_window=show_window,
            window_name=window_name
        )
        
        # 运行跟踪
        current_tracker.run_tracking(input_video_path)
        
        print(f"\n✓ 完成处理: {input_video_path}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 处理失败: {input_video_path}")
        print(f"错误信息: {str(e)}\n")
        return False
    finally:
        current_tracker = None

parser = argparse.ArgumentParser()
parser.add_argument("--weights",  type=str, default=ROOT / "weights/yolov10n-shangao-v3.pt", help="model path or triton URL")
parser.add_argument("--source", type=str, default=ROOT / "data/test/test.mp4", help="file/dir/URL/glob/screen/0(webcam)")
parser.add_argument("--output", type=str, default="output/", help="output path")
parser.add_argument("--show", action="store_true", help="显示实时可视化窗口")

args = parser.parse_args()

# 全局变量用于信号处理
current_tracker = None

# 捕获 Ctrl+C 信号
signal.signal(signal.SIGINT, signal_handler)

# 加载 YOLO 模型（只加载一次，供所有视频使用）
yolov10_model = YOLO(args.weights)

# 处理输入路径
input_path = str(args.source)
output_path = str(args.output)

# 检查输入是否为目录
if os.path.isdir(input_path):
    # 输入是目录，查找所有 MP4 文件
    print(f"检测到输入为目录: {input_path}")
    mp4_files = get_mp4_files(input_path)
    
    if not mp4_files:
        print(f"警告: 在目录 {input_path} 中未找到任何 MP4 文件")
        sys.exit(1)
    
    print(f"找到 {len(mp4_files)} 个 MP4 文件:")
    for i, f in enumerate(mp4_files, 1):
        print(f"  {i}. {f}")
    
    # 确保输出是目录
    if not os.path.isdir(output_path):
        print(f"警告: 输入是目录时，输出也应为目录。将使用输出路径的父目录")
        output_path = os.path.dirname(output_path) if os.path.dirname(output_path) else "output/"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # 遍历处理每个 MP4 文件
    success_count = 0
    fail_count = 0
    
    for idx, video_file in enumerate(mp4_files, 1):
        print(f"\n[{idx}/{len(mp4_files)}] 开始处理...")
        if process_single_video(yolov10_model, video_file, output_path, show_window=args.show):
            success_count += 1
        else:
            fail_count += 1
    
    # 统计检测到事件的视频
    videos_with_events = get_videos_with_events(output_path, mp4_files)
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"总计: {len(mp4_files)} 个文件")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"\n事件检测统计:")
    print(f"  检测到事件的视频总数: {len(videos_with_events)} 个")
    if videos_with_events:
        print(f"  检测到事件的视频文件名:")
        for i, filename in enumerate(videos_with_events, 1):
            print(f"    {i}. {filename}")
    else:
        print(f"  未检测到任何事件")
    print(f"{'='*60}\n")
    
else:
    # 输入是单个文件或流
    # 针对RTSP/RTMP/HTTP流或摄像头，生成带时间戳的文件名
    if os.path.isdir(output_path):
        if str(input_path).startswith(("rtsp://","rtmp://","http://","https://")) or str(input_path).isdigit():
            input_filename = f"stream_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        else:
            input_filename = os.path.basename(input_path)
        output_path = os.path.join(output_path, input_filename)
    
    print(f"最终输出路径: {output_path}")
    process_single_video(yolov10_model, input_path, output_path, show_window=args.show)
