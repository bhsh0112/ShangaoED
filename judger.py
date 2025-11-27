MIN_SPEED_WEIGHT=0.01
SLOW_SPEED_WEIGHT=0.2
MAX_JAM_VEHICLE_NUM=5
MIN_VEHICLE_WIDTH = 1.5
MAX_VEHICLE_GAP_WEIGHT = 2
STOP_DURATION_SECONDS = 3.0  # 速度连续低于阈值达到此秒数即认为停车




class Judger:
    def __init__(self,current_data,prev_data,result,jam_vehicle_info):
        self.current_data=current_data
        self.prev_data=prev_data
        self.result=result
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
        """计算阈值并更新拥堵候选列表。
        - 速度单位: 像素/秒；阈值按 fps 进行缩放
        - ROI: 若配置了 ROI，仅在区域内才参与拥堵候选
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

        #TODO:思考逻辑是否正确
        if self.result[0]==False:#jam
            if(self.isJam()):
                self.result[0]=True
                self.result[1]=False
            elif self.isParking():
                self.result[1]=True
        
        if self.isPeople():#people
            self.result[2]=True

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
        """
        if len(self.jam_vehicle_info) < MAX_JAM_VEHICLE_NUM:
            return False

        axis = 'x' if self.jam_axis not in ('x','y') else self.jam_axis
        sorted_vehicles = sorted(self.jam_vehicle_info, key=lambda v: v[axis])

        consecutive_count = 1
        for i in range(1, len(sorted_vehicles)):
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
                consecutive_count = 1

        return False
    
    def isPeople(self):
        return self.current_data['class']=="person"
    
    def is_slow_vehicle(self):
        """判断当前车辆是否为慢速车辆"""
        valid_classes = ["car", "truck", "bus", "motorcycle"]
        return (self.current_data['class'] in valid_classes and 
                self.current_data['speed'] < self.slow_speed)
