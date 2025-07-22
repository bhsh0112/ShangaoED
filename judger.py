MIN_SPEED_WEIGHT=0.01
MAX_JAM_VEHICLE_NUM=5
MIN_VEHICLE_WIDTH = 1.5



class Judger:
    def __init__(self,current_data,prev_data,result,jam_vehicle_num,interval):
        self.current_data=current_data
        self.prev_data=prev_data
        self.result=result
        self.jam_vehicle_num=jam_vehicle_num
        #TODO:合理的最小速度计算方法
        self.min_speed=self.current_data['size_w']*MIN_SPEED_WEIGHT
        # base_speed = max(current_data.get('size_w', MIN_VEHICLE_WIDTH), MIN_VEHICLE_WIDTH)
        # self.min_speed = base_speed * MIN_SPEED_WEIGHT

    def main(self):
        if (self.current_data['class']=="car" or self.current_data['class']=="truck") and self.current_data['speed']<self.min_speed:
            # print(self.jam_vehicle_num)
            self.jam_vehicle_num=self.jam_vehicle_num+1

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
                print(self.current_data['speed'])
                print(self.prev_data['speed'])
                print(self.min_speed)
                print("=======================")
                return self.current_data['speed']<=self.min_speed and self.prev_data['speed']<=self.min_speed
        else:
            return False
        
    def isJam(self):
        return self.jam_vehicle_num>=MAX_JAM_VEHICLE_NUM and self.current_data['speed']<=self.min_speed
    
    def isPeople(self):
        return self.current_data['class']=="person"
