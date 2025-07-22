MIN_SPEED_WEIGHT=0.01
SLOW_SPEED_WEIGHT=0.2
MAX_JAM_VEHICLE_NUM=5
MIN_VEHICLE_WIDTH = 1.5
MAX_VEHICLE_GAP_WEIGHT = 2




class Judger:
    def __init__(self,current_data,prev_data,result,jam_vehicle_info):
        self.current_data=current_data
        self.prev_data=prev_data
        self.result=result
        self.jam_vehicle_info = jam_vehicle_info
        #TODO:合理的最小速度计算方法
        self.min_speed=0
        self.slow_speed=0
        # base_speed = max(current_data.get('size_w', MIN_VEHICLE_WIDTH), MIN_VEHICLE_WIDTH)
        # self.min_speed = base_speed * MIN_SPEED_WEIGHT

    def main(self):
        self.min_speed=self.current_data['size_w']*MIN_SPEED_WEIGHT
        self.slow_speed=self.current_data['size_w']*SLOW_SPEED_WEIGHT
        if self.is_slow_vehicle():
            # 添加车辆位置信息（使用检测框中心点）
            vehicle_info = {
                'x': self.current_data['x'],
                'y': self.current_data['y'],
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
        # print(current_data['speed'])
        if(self.current_data['class']=="car" or self.current_data['class']=="truck"):
            if len(self.prev_data)==0:
                return False
            else:
                # print(self.current_data['speed'])
                # print(self.prev_data['speed'])
                # print(self.min_speed)
                # print("=======================")
                return self.current_data['speed']<=self.min_speed and self.prev_data['speed']<=self.min_speed
        else:
            return False
        
    def isJam(self):
        if len(self.jam_vehicle_info) < MAX_JAM_VEHICLE_NUM:
            return False
            
        # 按x坐标排序车辆
        sorted_vehicles = sorted(self.jam_vehicle_info, key=lambda v: v['x'])
        
        # 检查连续车辆
        consecutive_count = 1
        for i in range(1, len(sorted_vehicles)):
            prev = sorted_vehicles[i-1]
            curr = sorted_vehicles[i]
            
            # 计算车辆间距（欧氏距离）
            distance = ((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)**0.5
            print(distance)
            
            # 调整距离阈值（考虑车辆尺寸）
            gap_threshold = min(prev['width'], curr['width']) * MAX_VEHICLE_GAP_WEIGHT
            print(gap_threshold)
            print("==============================")
            
            if distance < gap_threshold:
                consecutive_count += 1
                if consecutive_count >= MAX_JAM_VEHICLE_NUM:
                    print("success!!!!!!")
                    return True
            else:
                consecutive_count = 1  # 重置连续计数
                
        return False
    
    def isPeople(self):
        return self.current_data['class']=="person"
    
    def is_slow_vehicle(self):
        """判断当前车辆是否为慢速车辆"""
        valid_classes = ["car", "truck", "bus", "motorcycle"]
        return (self.current_data['class'] in valid_classes and 
                self.current_data['speed'] < self.slow_speed)
