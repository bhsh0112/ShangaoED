MIN_SPEED_WEIGHT=1
MAX_JAM_VEHICLE_NUM=2

PARKED_MESSAGE="there are parked cars!"
JAM_MESSAGE="jam!"
PEOPLE_MESSAGE="there are people!"
NORMAL_MESSAGE="everything is ok"

EVENT_COLOR=(0,0,255)
NORMAL_COLOR=(0,255,0)

class Judger:
    def __init__(self,current_data,prev_data,prev_ED_message,jam_vehicle_num):
        self.current_data=current_data
        self.prev_data=prev_data
        self.prev_ED_message=prev_ED_message
        self.jam_vehicle_num=jam_vehicle_num
        #TODO:合理的最小速度计算方法
        self.min_speed=self.current_data['size_w']*MIN_SPEED_WEIGHT

    def main(self):
        if (self.current_data['class']=="car" or self.current_data['class']=="trunc") and self.current_data['speed']<self.min_speed:
            print(self.jam_vehicle_num)
            self.jam_vehicle_num=self.jam_vehicle_num+1

        #TODO:同时出现多种事件时
        if(self.prev_ED_message==NORMAL_MESSAGE):
            if self.isJam():
                print("success")
                return JAM_MESSAGE,EVENT_COLOR,self.jam_vehicle_num
            elif self.isParking():
                return PARKED_MESSAGE,EVENT_COLOR,self.jam_vehicle_num
            elif self.isPeople():
                return PEOPLE_MESSAGE,EVENT_COLOR,self.jam_vehicle_num
            else:
                return NORMAL_MESSAGE,NORMAL_COLOR,self.jam_vehicle_num
        else:
            return self.prev_ED_message,EVENT_COLOR,self.jam_vehicle_num

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
