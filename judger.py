MIN_SPEED_WEIGHT=1

PARKED_MESSAGE="there are parked cars!"
JAM_MESSAGE="jam!"
PEOPLE_MESSAGE="there are peple!"
NORMAL_MESSAGE="everything is ok"

EVENT_COLOR=(0,0,255)
NORMAL_COLOR=(0,255,0)

class Judger:
    def __init__(self,current_data,prev_data,prev_ED_message):
        self.current_data=current_data
        self.prev_data=prev_data
        self.prev_ED_message=prev_ED_message

    def main(self):
        #TODO:同时出现多种事件时
        if(self.prev_ED_message==NORMAL_MESSAGE):
            if self.isJam():
                return JAM_MESSAGE,EVENT_COLOR
            elif self.isParking():
                return PARKED_MESSAGE,EVENT_COLOR
            elif self.isPeople():
                return PEOPLE_MESSAGE,EVENT_COLOR
            else:
                return NORMAL_MESSAGE,NORMAL_COLOR 
        else:
            return self.prev_ED_message,EVENT_COLOR

    def isParking(self):
        #TODO:合理的最小速度计算方法
        min_speed=self.current_data['size_w']*MIN_SPEED_WEIGHT
        # print(current_data['speed'])
        if(self.current_data['class']=="car" or self.current_data['class']=="trunc"):
            if len(self.prev_data)==0:
                return False
            else:
                return self.current_data['speed']<=0.5 and self.prev_data['speed']<=0.5
        else:
            return False
        
    def isJam(self):
        #TODO
        return False
    
    def isPeople(self):
        return self.current_data['class']=="people"
        return False
