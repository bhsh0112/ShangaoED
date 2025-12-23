MIN_SPEED_WEIGHT=0.04
SLOW_SPEED_WEIGHT=0.25
MAX_JAM_VEHICLE_NUM=20
MIN_VEHICLE_WIDTH = 1.5
MAX_VEHICLE_GAP_WEIGHT = 3
STOP_DURATION_SECONDS = 3.0  # 速度连续低于阈值达到此秒数即认为停车
BREAKDOWN_DURATION_SECONDS = 180.0  # 停车时间超过此秒数（3分钟）即认为故障

# 车辆类别归一化因子：用于消除车辆类型对检测框尺寸的影响
# 假设在相同距离下，不同车辆类型的检测框尺寸比例
# 使用car作为参考（归一化因子=1.0），其他车辆类型根据典型尺寸比例调整
# 例如：truck在相同距离下的检测框约为car的1.2倍，则归一化因子为 1/1.2 ≈ 0.83
VEHICLE_NORMALIZATION_FACTOR = {
    'car': 1.0,           # 参考车辆，归一化因子为1.0
    'truck': 0.3,        # 卡车在相同距离下检测框更大，需要缩小
    'bus': 0.80,          # 公交车在相同距离下检测框更大，需要缩小
    'motorcycle': 1.3,    # 摩托车在相同距离下检测框更小，需要放大
    'default': 1.0        # 默认值
}

# 最小检测框尺寸阈值：小于此值的检测框可能距离过远，使用固定阈值
MIN_DETECTION_SIZE = 10.0
# 最大检测框尺寸阈值：大于此值的检测框可能距离过近，使用固定阈值
MAX_DETECTION_SIZE = 200.0

# 基于距离的速度阈值计算参数
# 距离因子范围：用于根据目标框距离图像底部的距离调整速度阈值
# 图像底部（y 接近 image_height）代表近距离，使用最大因子
# 图像顶部（y 接近 0）代表远距离，使用最小因子
MIN_DISTANCE_FACTOR = 0.3  # 最远距离时的最小距离因子（阈值会更小）
MAX_DISTANCE_FACTOR = 1.0  # 最近距离时的最大距离因子（阈值会更大）
# 基础速度阈值：用于乘以距离因子得到最终阈值
BASE_MIN_SPEED_THRESHOLD = 15.0  # 停车判断的基础阈值（像素/帧）
BASE_SLOW_SPEED_THRESHOLD = 50.0  # 拥堵判断的基础阈值（像素/帧）




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
        # 图像高度（用于计算目标框距离底部的距离）
        self.image_height = getattr(self, 'image_height', None)
        # 计算合理的最小速度阈值
        self.min_speed = self._calculate_min_speed()
        self.slow_speed = self._calculate_slow_speed()

    def main(self):
        """处理单个车辆数据并更新拥堵候选列表。
        - 速度单位: 像素/秒；阈值按 fps 进行缩放
        - ROI: 若配置了 ROI，仅在区域内才参与拥堵候选
        - 注意：拥堵判断需要在处理完当前帧所有车辆后，通过 checkJam() 方法调用
        """
        # 如果 current_data 为 None，直接返回
        if self.current_data is None:
            return
        
        # 重新计算速度阈值（因为 current_data 可能已更新）
        self.min_speed = self._calculate_min_speed()
        self.slow_speed = self._calculate_slow_speed()
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
            # 根据当前帧的停车判断结果更新停车状态
            # 如果当前速度高于阈值，isParking() 会返回 False，需要重置停车状态
            if self.isParking():
                self.result[1]=True
                # 检查是否达到故障条件（停车时间超过3分钟）
                if self.isBreakdown():
                    self.result[3]=True
            else:
                # 当前帧不满足停车条件，重置停车状态（但保留故障状态，因为故障是累积的）
                self.result[1]=False
        
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

        # 只有当当前速度低于阈值且累计持续时间满足条件时，才判定为停车
        # 如果当前速度高于阈值，即使历史持续时间很长，也不应该判定为停车
        return is_low_speed and current_duration >= STOP_DURATION_SECONDS
        
    def isJam(self):
        """拥堵判定
        - 先按配置轴 (x/y) 对慢速车辆排序
        - 使用沿该轴的一维间距与车辆宽度成比例的阈值判断是否紧邻
        - 存在长度 ≥ MAX_JAM_VEHICLE_NUM 的紧邻序列则认为候选拥堵
        
        Returns:
            bool: 如果检测到拥堵返回 True，否则返回 False
        """
        if len(self.jam_vehicle_info) < MAX_JAM_VEHICLE_NUM:
            return False

        # 确定排序轴，默认为 'x'
        axis = 'x' if self.jam_axis not in ('x', 'y') else self.jam_axis
        
        # 按指定轴排序
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
    
    def _calculate_min_speed(self):
        """
        计算合理的最小速度阈值（停车判断用）
        
        基于目标框距离图像底部的距离计算速度阈值：
        1. 目标框中心点的 y 坐标越大，距离图像底部越近，距离摄像头越近
        2. 距离摄像头越近，速度阈值应该越大（因为相同的像素速度对应的实际速度更大）
        3. 距离摄像头越远，速度阈值应该越小（因为相同的像素速度对应的实际速度更小）
        
        计算逻辑：
        - 计算距离因子：distance_factor = min_factor + (max_factor - min_factor) * (y / image_height)
        - 速度阈值 = base_threshold * distance_factor
        
        核心思想：
        - 高速公路场景中，摄像头通常安装在较高位置，视野向下
        - 图像底部（y 接近 image_height）代表近距离，阈值应该更大
        - 图像顶部（y 接近 0）代表远距离，阈值应该更小
        
        Returns:
            float: 最小速度阈值（像素/帧，未乘以fps）
        """
        # 如果 current_data 为 None，返回默认值
        if self.current_data is None:
            return BASE_MIN_SPEED_THRESHOLD * MIN_DISTANCE_FACTOR
        
        # 获取目标框中心点的 y 坐标
        y = self.current_data.get('y', None)
        
        # 如果没有图像高度信息或 y 坐标，使用默认距离因子
        if self.image_height is None or y is None or self.image_height <= 0:
            # 如果没有距离信息，使用中等距离因子
            distance_factor = (MIN_DISTANCE_FACTOR + MAX_DISTANCE_FACTOR) / 2.0
            return BASE_MIN_SPEED_THRESHOLD * distance_factor
        
        # 计算归一化的 y 坐标位置（0 到 1 之间）
        # y 越大，说明越靠近图像底部，距离摄像头越近
        normalized_y = y / float(self.image_height)
        
        # 限制在 [0, 1] 范围内
        normalized_y = max(0.0, min(1.0, normalized_y))
        
        # 计算距离因子：线性插值
        # normalized_y = 0 (顶部，远距离) -> 使用 MIN_DISTANCE_FACTOR
        # normalized_y = 1 (底部，近距离) -> 使用 MAX_DISTANCE_FACTOR
        distance_factor = MIN_DISTANCE_FACTOR + (MAX_DISTANCE_FACTOR - MIN_DISTANCE_FACTOR) * normalized_y
        
        # 基于距离因子计算速度阈值
        return BASE_MIN_SPEED_THRESHOLD * distance_factor
    
    def _calculate_slow_speed(self):
        """
        计算慢速车辆阈值（拥堵判断用）
        
        使用与停车判断相同的基于距离的计算逻辑，确保一致性
        
        基于目标框距离图像底部的距离计算速度阈值：
        - 距离摄像头越近，速度阈值应该越大
        - 距离摄像头越远，速度阈值应该越小
        
        Returns:
            float: 慢速阈值（像素/帧，未乘以fps）
        """
        # 如果 current_data 为 None，返回默认值
        if self.current_data is None:
            return BASE_SLOW_SPEED_THRESHOLD * MIN_DISTANCE_FACTOR
        
        # 获取目标框中心点的 y 坐标
        y = self.current_data.get('y', None)
        
        # 如果没有图像高度信息或 y 坐标，使用默认距离因子
        if self.image_height is None or y is None or self.image_height <= 0:
            # 如果没有距离信息，使用中等距离因子
            distance_factor = (MIN_DISTANCE_FACTOR + MAX_DISTANCE_FACTOR) / 2.0
            return BASE_SLOW_SPEED_THRESHOLD * distance_factor
        
        # 计算归一化的 y 坐标位置（0 到 1 之间）
        # y 越大，说明越靠近图像底部，距离摄像头越近
        normalized_y = y / float(self.image_height)
        
        # 限制在 [0, 1] 范围内
        normalized_y = max(0.0, min(1.0, normalized_y))
        
        # 计算距离因子：线性插值
        # normalized_y = 0 (顶部，远距离) -> 使用 MIN_DISTANCE_FACTOR
        # normalized_y = 1 (底部，近距离) -> 使用 MAX_DISTANCE_FACTOR
        distance_factor = MIN_DISTANCE_FACTOR + (MAX_DISTANCE_FACTOR - MIN_DISTANCE_FACTOR) * normalized_y
        
        # 基于距离因子计算速度阈值
        return BASE_SLOW_SPEED_THRESHOLD * distance_factor
    
    def is_slow_vehicle(self):
        """判断当前车辆是否为慢速车辆"""
        valid_classes = ["car", "truck", "bus", "motorcycle"]
        return (self.current_data['class'] in valid_classes and 
                self.current_data['speed'] < self.slow_speed)
