MIN_SPEED_WEIGHT=1
MAX_JAM_VEHICLE_NUM=2



class Judger:
    def __init__(self,current_data,prev_data,result,jam_vehicle_num):
        self.current_data=current_data
        self.prev_data=prev_data
        self.result=result
        self.jam_vehicle_num=jam_vehicle_num
        #TODO:合理的最小速度计算方法
        self.min_speed=self.current_data['size_w']*MIN_SPEED_WEIGHT

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
        if(self.current_data['class']=="car" or self.current_data['class']=="trunc"):
            if len(self.prev_data)==0:
                return False
            else:
                return self.current_data['speed']<=0.5 and self.prev_data['speed']<=0.5
        else:
            return False
        
    def isJam(self):
        return self.jam_vehicle_num>=MAX_JAM_VEHICLE_NUM
    
    def isPeople(self):
        return self.current_data['class']=="person"
