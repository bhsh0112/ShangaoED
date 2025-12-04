MIN_SPEED_WEIGHT=0.03
SLOW_SPEED_WEIGHT=1
MAX_JAM_VEHICLE_NUM=10
MIN_VEHICLE_WIDTH = 1.5
MAX_VEHICLE_GAP_WEIGHT = 2
STOP_DURATION_SECONDS = 2.0  # 速度连续低于阈值达到此秒数即认为停车
BREAKDOWN_DURATION_SECONDS = 180.0  # 停车时间超过此秒数（3分钟）即认为故障




class Judger:
    def __init__(self,current_data,prev_data,result,jam_vehicle_info):
        self.current_data=current_data
        self.prev_data=prev_data
        # 确保result列表至少有4个元素 [jam, park, people, breakdown]
        if result is None:
            self.result = [False, False, False, False]
        else:
            self.result = list(result)
            # 如果result少于4个元素，补齐到4个
            while len(self.result) < 4:
                self.result.append(False)
        self.jam_vehicle_info = jam_vehicle_info
        # 方向与区域配置（由外部注入，默认为全局与按 x 轴）
        self.jam_axis = getattr(self, 'jam_axis', 'x')
        self.roi = getattr(self, 'roi', None)  # (x1, y1, x2, y2)
        #TODO:合理的最小速度计算方法
        self.min_speed=0
        self.slow_speed=0
        # base_speed = max(current_data.get('size_w', MIN_VEHICLE_WIDTH), MIN_VEHICLE_WIDTH)
        # self.min_speed = base_speed * MIN_SPEED_WEIGHT

    def main(self):
        """处理单个车辆数据并更新拥堵候选列表。
        - 速度单位: 像素/秒；阈值按 fps 进行缩放
        - ROI: 若配置了 ROI，仅在区域内才参与拥堵候选
        - 注意：拥堵判断需要在处理完当前帧所有车辆后，通过 checkJam() 方法调用
        """
        self.min_speed=self.current_data['size_w']*MIN_SPEED_WEIGHT
        self.slow_speed=self.current_data['size_w']*SLOW_SPEED_WEIGHT
        fps = float(self.current_data.get('fps', 0) or 0)
        if fps > 0:
            self.min_speed *= fps
            self.slow_speed *= fps
        if self.is_slow_vehicle():
            x = self.current_data['x']
            y = self.current_data['y']
            in_roi = True
            if self.roi is not None:
                x1, y1, x2, y2 = self.roi
                in_roi = (x >= x1 and x <= x2 and y >= y1 and y <= y2)
            if in_roi:
                # 添加车辆位置信息（使用检测框中心点）
                vehicle_info = {
                    'x': x,
                    'y': y,
                    'width': self.current_data['size_w']
                }
                self.jam_vehicle_info.append(vehicle_info)

        # 处理单个车辆的事件判断（停车、故障和行人）
        # 注意：拥堵判断需要基于当前帧所有车辆，应在处理完所有车辆后调用 checkJam()
        if self.result[0]==False:#jam
            if self.isParking():
                self.result[1]=True
                # 检查是否达到故障条件（停车时间超过3分钟）
                if self.isBreakdown():
                    self.result[3]=True
        
        if self.isPeople():#people
            self.result[2]=True
    
    def checkJam(self):
        """检查当前帧是否存在拥堵
        应该在处理完当前帧所有车辆后调用此方法
        
        Returns:
            bool: 如果检测到拥堵返回 True，否则返回 False
        """
        if self.isJam():
            self.result[0] = True
            self.result[1] = False  # 拥堵时不是停车
            return True
        return False

    def isParking(self):
        # 使用持续时间判定停车：当速度连续低于阈值达到 STOP_DURATION_SECONDS
        valid_vehicle = (self.current_data['class'] == "car" or self.current_data['class'] == "truck")
        if not valid_vehicle:
            return False

        if len(self.prev_data) == 0:
            # 首帧，无历史数据，初始化持续时长
            self.current_data['low_speed_duration'] = 0.0
            return False

        # 基于尺寸的最小速度阈值（与原逻辑一致）
        speed_threshold = self.min_speed

        # 计算时间增量（秒），优先使用帧差/帧率
        current_frame = self.current_data.get('frame', None)
        prev_frame = self.prev_data.get('frame', None)
        fps = self.current_data.get('fps', None)
        if current_frame is not None and prev_frame is not None and fps and fps > 0:
            dt = (current_frame - prev_frame) / float(fps)
        else:
            prev_time = self.prev_data.get('Time', self.current_data['Time'])
            dt = self.current_data['Time'] - prev_time
        if dt < 0:
            dt = 0.0

        # 累计低速持续时间
        prev_duration = float(self.prev_data.get('low_speed_duration', 0.0))
        is_low_speed = self.current_data['speed'] <= speed_threshold
        current_duration = prev_duration + dt if is_low_speed else 0.0
        self.current_data['low_speed_duration'] = current_duration

        return current_duration >= STOP_DURATION_SECONDS
        
    def isJam(self):
        """拥堵判定
        - 先按配置轴 (x/y) 对慢速车辆排序
        - 使用沿该轴的一维间距与车辆宽度成比例的阈值判断是否紧邻
        - 存在长度 ≥ MAX_JAM_VEHICLE_NUM 的紧邻序列则认为候选拥堵
        
        Returns:
            bool: 如果检测到拥堵返回 True，否则返回 False
        """
        print(len(self.jam_vehicle_info))
        if len(self.jam_vehicle_info) < MAX_JAM_VEHICLE_NUM:
            return False

        # 确定排序轴，默认为 'x'
        axis = 'x' if self.jam_axis not in ('x', 'y') else self.jam_axis
        
        # 按指定轴排序
        sorted_vehicles = sorted(self.jam_vehicle_info, key=lambda v: v[axis])

        consecutive_count = 1
        for i in range(1, len(sorted_vehicles)):
            print("===================debug================")
            
            print(consecutive_count)
            prev = sorted_vehicles[i-1]
            curr = sorted_vehicles[i]

            # 仅比较轴向间距，更符合排队方向
            axis_distance = abs(curr[axis] - prev[axis])
            gap_threshold = min(prev['width'], curr['width']) * MAX_VEHICLE_GAP_WEIGHT

            if axis_distance < gap_threshold:
                consecutive_count += 1
                if consecutive_count >= MAX_JAM_VEHICLE_NUM:
                    return True
            else:
                # 重置连续计数，当前车辆作为新序列的开始
                consecutive_count = 1

        return False
    
    def isPeople(self):
        """
        判断当前检测对象是否为行人
        需要同时满足：类别为person且置信度达到0.8
        
        Returns:
            bool: 如果是行人且置信度>=0.8返回True，否则返回False
        """
        return (self.current_data['class'] == "person" and 
                self.current_data.get('confidence', 0.0) >= 0.8)
    
    def isBreakdown(self):
        """
        判断当前车辆是否为故障（停车时间超过3分钟）
        需要同时满足：车辆处于停车状态且停车持续时间超过BREAKDOWN_DURATION_SECONDS
        
        Returns:
            bool: 如果停车时间超过3分钟返回True，否则返回False
        """
        # 只有在停车状态下才可能发生故障
        if not self.isParking():
            return False
        
        # 获取当前的低速持续时间（已在isParking中计算）
        current_duration = float(self.current_data.get('low_speed_duration', 0.0))
        
        return current_duration >= BREAKDOWN_DURATION_SECONDS
    
    def is_slow_vehicle(self):
        """判断当前车辆是否为慢速车辆"""
        valid_classes = ["car", "truck", "bus", "motorcycle"]
        return (self.current_data['class'] in valid_classes and 
                self.current_data['speed'] < self.slow_speed)
