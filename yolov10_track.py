class YOLOv10Tracker:
    def __init__(self, yolov10_model):
        if not yolov10_model:
            raise ValueError("YOLOv10 model cannot be None")
        self.model = yolov10_model
        self.tracked_data = []
        self.frame = []
        self.detection_interval = 1  # 统计间隔（秒）
        self.last_stat_time = time.time()
        global global_a_count  # 声明使用全局变量
        global_a_count = 0  # 修改全局变量
        self.vehicle_data = pd.DataFrame(columns=[
            'id', 'Time', 'type', 'x', 'y', 'xv', 'yv', 
            'yaw', 'length', 'width'
        ])
        # speed_obj = speed_estimation.SpeedEstimator()


    def get_frame(self):
        return self.frame

    def generate_statistics(self, frame):
        """生成并显示统计信息"""
        if not self.vehicle_data.empty:
            # 示例统计区域（需要根据实际坐标调整）
            AREA = (0, 1920, 0, 1080)  # x1, x2, y1, y2
            # 调用统计函数
            a_count = count_a_class_vehicles_in_area(self.vehicle_data, *AREA)
            non_motor_count = count_non_motorized_vehicles(self.vehicle_data)
            avg_speed = calculate_average_speed(self.vehicle_data)

            # 在画面左上角显示统计信息
            text_y = 50
            cv2.putText(frame, f"Category A Vehicles: {a_count}", (20, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y += 40
            cv2.putText(frame, f"Non-motor Vehicles: {non_motor_count}", (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_y += 40
            cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return a_count, non_motor_count, avg_speed
        else:
            # 车辆数据为空时返回默认值
            return global_a_count,global_non_motor_count,global_avg_speed 


    def run_tracking(self, url):
        global global_a_count
        global_a_count = 0
        """运行YOLOv10模型进行目标检测"""
        print(f"开始处理视频流：{url}")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 只保留最新的1帧
        assert cap.isOpened(), "非法的视频文件或无法打开视频流"
        while cap.isOpened():
                        # 丢弃旧帧，读取最新的
            for _ in range(2):  # 丢弃最近的 2 帧
                ret, self.frame = cap.read()
            if not ret:
                break
                        # 数据收集逻辑
            current_time = time.time()
            if current_time - self.last_stat_time >= self.detection_interval:
                self.last_stat_time = current_time
                global_a_count,global_non_motor_count,global_avg_speed = self.generate_statistics(self.frame)
                self.vehicle_data = pd.DataFrame()  # 清空数据
            else:
                # 在画面左上角显示统计信息
                text_y = 50
                cv2.putText(self.frame, f"Category A Vehicles: {global_a_count}", (20, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                text_y += 40
                cv2.putText(self.frame, f"Non-motor Vehicles: {global_non_motor_count}", (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                text_y += 40
                cv2.putText(self.frame, f"Avg Speed: {global_avg_speed:.1f} km/h", (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            matched_radar_data = None
            # 加入时间戳并将数据放入队列
            timestamp = time.time()
            while not radar_queue.empty():
                radar_item = radar_queue.get()
                radar_timestamp = radar_item["timestamp"]
                delta_time = abs(timestamp - radar_timestamp)
                if delta_time <=0.36:
                    matched_radar_data = radar_item
                    print(f'time matched!!! delta_time: {delta_time:.2f}')
                    break
                else:
                    print(f'time unmatched!!! delta_time: {delta_time:.2f}')
            if matched_radar_data:
                radar_data = matched_radar_data["pixel"]
                for i in range(len(radar_data)):
                    radar_x, radar_y = radar_data[i]
                    # print('radar_x:',radar_x,' radar_y:',radar_y)
                    cv2.circle(self.frame, (radar_x, radar_y), 3, (0, 0, 255), -1)
                    data = matched_radar_data["data"][i]
                    dis_x,dix_y = data['X Distance (m)'],data['Y Distance (m)']
                    vx,vy = data['Speed X (km/h)'],data['Speed Y (km/h)']
                    id = data['Target ID']
                    cv2.putText(self.frame, f"X:{dis_x:.1f} Y:{dix_y:.1f}", (radar_x, radar_y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    # 计算宽度和高度
                    box_height,box_width = calculate_box_size(radar_y)
                    # print('box_width:',box_width,'box_height:',box_height)
                    # print('vx:',vx,'vy:',vy)
                    top_left2 = (radar_x - box_width // 2, radar_y - box_height // 2)
                    bottom_right2 = (radar_x + box_width // 2, radar_y + box_height // 2)
                    # cv2.putText(camera_frame, f" X:{dis_x:.1f} Y:{dix_y:.1f}", (top_left2[0], top_left2[1] -10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    cv2.putText(self.frame, f"ID:{id} vx:{vx:.1f} vy:{vy:.1f}", (top_left2[0], top_left2[1] -10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    # 绘制检测框
                    cv2.rectangle(self.frame, top_left2, bottom_right2, (0, 0, 255), 2)
                    # new_data = pd.DataFrame([{
                    #     'id': id,
                    #     'Time': timestamp,
                    #     'type': 2, 
                    #     'x': dis_x,
                    #     'y': dix_y,
                    #     'xv': vx,
                    #     'yv': vy,
                    #     'yaw': 0,  # 需要实际航向角数据
                    #     'length': 6,  # 示例换算比例
                    #     'width': 3
                    # }])
                    # self.vehicle_data = pd.concat([self.vehicle_data, new_data])




            # 执行YOLOv10目标检测
            # results = self.model(self.frame)  # 使用已加载的模型进行推断
            tracks = self.model.track(self.frame, conf=0.1, iou=0.1,persist=True, show=False,verbose=False)
            self.frame = counter_obj.start_counting(self.frame, tracks)
            self.frame = speed_obj.estimate_speed(self.frame, tracks)
            for result in tracks:
                # 获取每个结果的边界框、置信度和类别
                boxes = result.boxes  # 获取边界框数据
                confidences = boxes.conf  # 获取置信度数据
                class_ids = boxes.cls  # 获取类别索引数据
                # print(result)

                # 遍历每个检测框
                for i, box in enumerate(boxes.xyxy):
                    x_min, y_min, x_max, y_max = box  # 获取框的坐标
                    confidence = confidences[i]  # 获取当前框的置信度
                    class_id = int(class_ids[i])  # 获取类别 ID
                    class_name = result.names[class_id]  # 获取类别名称
                    id = result[-1] # 或者可能是 result.track_ids

                    if confidence < 0.1:
                        continue

                    # 设定边界框颜色，通常我们会用 BGR 格式
                    color = (0, 255, 0)  # 绿色框（可以根据类别调整颜色）

                    # 绘制类别和置信度
                    label = f"{class_name}: {confidence:.2f}"  # 标签格式：类别：置信度
                    best_iou = 0
                    best_radar_box = None
                    best_radar_data = None

                    dis_x = 0
                    dis_y = 0
                    vx = 0
                    vy = 0
                    # 遍历毫米波目标框
                    if matched_radar_data:
                        radar_data = matched_radar_data["pixel"]
                        for radar_point in radar_data:
                            radar_x, radar_y = radar_point

                            # 计算毫米波目标框
                            box_height, box_width = calculate_box_size(radar_y)
                            radar_box = (
                                radar_x - box_width // 2, radar_y - box_height // 2,
                                radar_x + box_width // 2, radar_y + box_height // 2
                            )

                            # 计算 IoU
                            iou = compute_iou(box, radar_box)
                            if iou > 0:
                                best_iou = iou
                                best_radar_box = radar_box
                                best_radar_data = matched_radar_data["data"][radar_data.index(radar_point)]
                                # 计算 YOLO 框的中心点
                                yolo_center_x = (x_min + x_max) / 2
                                yolo_center_y = (y_min + y_max) / 2

                                # 增大 YOLO 框作为融合框
                                merged_x1 = int(x_min - 10)  # 向左扩展
                                merged_y1 = int(y_min - 10)  # 向上扩展
                                merged_x2 = int(x_max + 10)  # 向右扩展
                                merged_y2 = int(y_max + 10)  # 向下扩展
                                merged_bbox = (merged_x1, merged_y1, merged_x2, merged_y2)

                                # 调整毫米波框，使其小于 YOLO 框，并保持对齐
                                radar_box_center_x = (best_radar_box[0] + best_radar_box[2]) / 2
                                radar_box_center_y = (best_radar_box[1] + best_radar_box[3]) / 2

                                # 将毫米波框中心对齐于 YOLO 框的中心
                                new_radar_box_x1 = int(yolo_center_x - (x_max - x_min) // 4)
                                new_radar_box_y1 = int(yolo_center_y - (y_max - y_min) // 4)
                                new_radar_box_x2 = int(yolo_center_x + (x_max - x_min) // 4)
                                new_radar_box_y2 = int(yolo_center_y + (y_max - y_min) // 4)

                                # 绘制 YOLO 框
                                cv2.rectangle(self.frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                                cv2.putText(self.frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                # 绘制融合框
                                cv2.rectangle(self.frame, (merged_x1, merged_y1), (merged_x2, merged_y2), (0, 255, 255), 2)
                                cv2.putText(self.frame, label, (merged_x1, merged_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                                # 绘制毫米波框
                                # cv2.rectangle(self.frame, (new_radar_box_x1, new_radar_box_y1), (new_radar_box_x2, new_radar_box_y2), (0, 0, 255), 2)

                                # 显示毫米波速度 & 距离信息
                                dis_x, dis_y = best_radar_data['X Distance (m)'], best_radar_data['Y Distance (m)']
                                vx, vy = best_radar_data['Speed X (km/h)'], best_radar_data['Speed Y (km/h)']
                                # target_id = best_radar_data['Target ID']

                                # cv2.putText(self.frame, f"ID:{target_id} vx:{vx:.1f} vy:{vy:.1f}", (merged_x1, merged_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                # cv2.putText(self.frame, f"X:{dis_x:.1f} Y:{dis_y:.1f}", (merged_x1, merged_y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                break
                    width, length = get_object_size(class_name)
                    new_data = pd.DataFrame([{
                        'id': id,
                        'Time': current_time,
                        'type': class_id, 
                        'x': dis_x,
                        'y': dis_y,
                        'xv': vx,
                        'yv': vy,
                        'yaw': 0,  # 需要实际航向角数据
                        'length': length,  # 示例换算比例
                        'width': width
                    }])
                    self.vehicle_data = pd.concat([self.vehicle_data, new_data])
            yolo_queue.put({"type": "YOLOv10", "data": self.tracked_data, "timestamp": timestamp, "frame": self.frame})
            self.tracked_data = []

        cap.release()

class RadarProcessor:
    def __init__(self, host="192.168.1.88", port=5000):
        self.host = host
        self.port = port
        self.radar_data = []
        self.running = True
        self.data_lock = threading.Lock()  # 锁用于线程同步bv 

        self.h_matrix_tmp = np.array([  [ 5.78063865e-01,  5.77392301e-01, -7.10299161e+02], 
                                        [ 7.35310132e-01,  2.37751074e+00, -2.12225704e+02], 
                                        [-1.42923517e-03,  1.35059261e-01,  1.00000000e+00]])
        # 启动接收数据的线程
        self.receiver_thread = threading.Thread(target=self.start_receiving)
        self.receiver_thread.start()

    def send_message(self, message):
        """发送消息到雷达服务器"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            s.sendall(b"ADD-ASCII " + message.encode('ascii'))
            print("Message sent.")
        except Exception as e:
            print("Error sending message:", e)
        finally:
            s.close()
    def generate_bounding_boxes(self):
        """根据聚类结果生成二维框"""
        cluster_boxes = {}
        
        # 遍历雷达数据，将每个聚类组的目标数据合并
        for data in self.radar_data:
            pos_cluster_id = data.get("Position Cluster ID")
            if pos_cluster_id not in cluster_boxes:
                cluster_boxes[pos_cluster_id] = {
                    "min_x": float('inf'),
                    "max_x": float('-inf'),
                    "min_y": float('inf'),
                    "max_y": float('-inf')
                }
            
            # 更新该聚类框的边界
            cluster_boxes[pos_cluster_id]["min_x"] = min(cluster_boxes[pos_cluster_id]["min_x"], data["X Distance (m)"])
            cluster_boxes[pos_cluster_id]["max_x"] = max(cluster_boxes[pos_cluster_id]["max_x"], data["X Distance (m)"])
            cluster_boxes[pos_cluster_id]["min_y"] = min(cluster_boxes[pos_cluster_id]["min_y"], data["Y Distance (m)"])
            cluster_boxes[pos_cluster_id]["max_y"] = max(cluster_boxes[pos_cluster_id]["max_y"], data["Y Distance (m)"])
        
        # 输出生成的二维框
        bounding_boxes = []
        for cluster_id, box in cluster_boxes.items():
            bounding_boxes.append({
                "Cluster ID": cluster_id,
                "Bounding Box": {
                    "min_x": box["min_x"],
                    "max_x": box["max_x"],
                    "min_y": box["min_y"],
                    "max_y": box["max_y"]
                }
            })
        
        return bounding_boxes
    def cluster_radar_data(self, radar_points):
        """使用 DBSCAN 进行位置聚类"""
        if len(radar_points) < 2:
            print("数据点不足，无法聚类。")
            return

        # 检查并去除 NaN 数据点
        radar_points = np.array(radar_points)
        radar_points = radar_points[~np.isnan(radar_points).any(axis=1)]  # 删除包含 NaN 的行
        
        if len(radar_points) < 2:
            print("数据点不足，无法聚类。")
            return
        
        # 使用 DBSCAN 进行位置聚类
        dbscan = DBSCANManual(eps=5, min_samples=2)
        labels = dbscan.fit(radar_points)

        print("位置聚类结果：", labels)
        
        # 将位置聚类结果添加到雷达数据中
        for i, label in enumerate(labels):
            self.radar_data[i]["Position Cluster ID"] = label  # 给每个雷达目标打上位置聚类标签
    
    def cluster_velocity_data(self, velocity_points):
        """使用 DBSCAN 进行速度聚类"""
        if len(velocity_points) < 2:
            print("速度数据点不足，无法聚类。")
            return
        
        # 使用 DBSCAN 进行速度聚类
        dbscan = DBSCANManual(eps=5, min_samples=2)
        labels = dbscan.fit(velocity_points)

        print("速度聚类结果：", labels)
        
        # 将速度聚类结果添加到雷达数据中
        for i, label in enumerate(labels):
            self.radar_data[i]["Velocity Cluster ID"] = label  # 给每个雷达目标打上速度聚类标
    def receive_message(self):
        """从雷达服务器接收消息"""
        while self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, self.port))
                print("Receiving messages:")
                while self.running:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    self.parse_frame(chunk)
            except Exception as e:
                print("Error receiving message:", e)
                try:
                    file_content = 'REST'
                    self.send_message(file_content)
                except Exception as e:
                    print("Error:", e)
            time.sleep(1)

    def parse_frame(self, data):
        """解析帧数据"""
        self.radar_data = []
        header_size = 4  # 帧头 "TRAJ" 占 4 字节
        frame_length_size = 4  # 帧长 4 字节

        # 读取帧头
        frame_header = data[:header_size].decode('ascii')
        if frame_header != "TRAJ":
            return

        # 读取帧长
        frame_length = int.from_bytes(data[header_size:header_size + frame_length_size], byteorder='big')

        # 计算目标数量 N
        if frame_length >= 74 and (frame_length - 74) % 80 == 0:
            N = (frame_length - 74) // 80
        else:
            raise ValueError("Invalid frame length. Cannot compute target count N.")
        frame_data = data[header_size + frame_length_size:]
        self.parse_target_data(frame_data, N)

    def radar_to_camera(self,radar_points):
        """
        将毫米波雷达的 x, y 坐标点映射到相机像素点。

        参数:
        radar_points (list of tuples): 雷达坐标点列表，每个点为 (x, y)
        perspective_matrix (numpy.ndarray): 3x3 的透视变换矩阵

        返回:
        camera_points (list of tuples): 相机像素坐标点列表，每个点为 (x, y)
        """
        if radar_points:
            camera_points = []
            for x, y in radar_points:
                # 将雷达点（毫米波坐标）转换为齐次坐标
                point_3D_01 = np.array([x, y, 1], dtype=np.float32)

                # 使用单应矩阵计算像素坐标
                pixel_location_3d = np.linalg.inv(self.h_matrix_tmp) @ point_3D_01

                # 计算二维像素点的坐标
                center_x = pixel_location_3d[0] / pixel_location_3d[2]
                center_y = pixel_location_3d[1] / pixel_location_3d[2]
                # print('x',x,'y',y,'cx',center_x,'cy',center_y)
                # 添加到结果列表
                camera_points.append((int(center_x), int(center_y)))

            return camera_points
        else:
            return []


    def parse_target_data(self, data, num_targets):
        """解析目标数据并进行处理"""
        target_data_list = []
        radar_points = []
        velocity_points = []  # 用于存储速度信息
        # 初始字节偏移量，第一个目标的ID从72字节开始
        offset = 72

        for i in range(num_targets):
            # 解析目标ID号 (1 byte uint8)
            target_id = struct.unpack_from('>B', data, offset)[0]
            offset += 1

            # 解析X距离 (4 bytes float, 大端模式)
            x_distance = struct.unpack_from('>f', data, offset)[0]
            offset += 4

            # 解析Y距离 (4 bytes float, 大端模式)
            y_distance = struct.unpack_from('>f', data, offset)[0]
            offset += 4

            # 解析速度_X (4 bytes float, 大端模式)
            speed_x = struct.unpack_from('>f', data, offset)[0]
            offset += 4

            # 解析速度_Y (4 bytes float, 大端模式)
            speed_y = struct.unpack_from('>f', data, offset)[0]
            offset += 4

            # 跳过其余目标数据 (80字节每目标 - 已读取的字段)
            offset += 80 - (1 + 4 + 4 + 4 + 4)
            if x_distance > -5 and y_distance > 30:

                target_data = {
                    "Target ID": target_id,
                    "X Distance (m)": x_distance,
                    "Y Distance (m)": y_distance,
                    "Speed X (km/h)": speed_x,
                    "Speed Y (km/h)": speed_y
                }
                # print("X:",x_distance,' Y:',y_distance)

                radar_points.append((x_distance, y_distance))
                velocity_points.append((speed_x, speed_y))
                target_data_list.append(target_data)

        self.radar_data.extend(target_data_list)
        # 先进行位置聚类
        # self.cluster_radar_data(radar_points)
        # 然后根据速度信息进行聚类
        # self.cluster_velocity_data(velocity_points)
        # 加入时间戳并将数据放入队列
        timestamp = time.time()
        # radar_position = []
        # for target in self.radar_data:
        #     item = (target["X Distance (m)"], target["Y Distance (m)"])
        #     radar_position.append(item)
        radar_position = list((target["X Distance (m)"], target["Y Distance (m)"]) for target in self.radar_data)
        camera_points = self.radar_to_camera(radar_position)
        # camera_points = [(888,555),(655,744)]
        # print('camera_point',camera_points)
        # data_queue.put({"type": "Radar", "data": self.radar_data, "pixel": camera_points, "timestamp": timestamp})
        if camera_points:
            print('pixel points:',camera_points)
            radar_queue.put({"type": "Radar", "data": self.radar_data, "pixel": camera_points, "timestamp": timestamp})

    def get_target_data(self):
        with self.data_lock:
            return self.radar_data

    def start_receiving(self):
        """持续接收并解析雷达数据"""
        print("Starting radar data processing...")
        self.receive_message()

    def stop_receiving(self):
        """停止接收数据"""
        self.running = False
        self.receiver_thread.join()
        print("接收线程已停止")

    

    yolo_tracker = YOLOv10Tracker(yolov10_model)  