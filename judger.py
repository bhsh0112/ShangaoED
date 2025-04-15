MIN_SPEED_WEIGHT=1

PARKED_MESSAGE="there are parked cars!"
JAM_MESSAGE="jam!"

EVENT_COLOR=(0,0,255)
NORMAL_COLOR=(0,255,0)

class Judger:
    def __init__(self,current_data,prev_data):
        self.current_data=current_data
        self.prev_data=prev_data

    def main():
        if
        return 

    def isParking(self,current_data,prev_data):
        
        min_speed=current_data['size_w']*MIN_SPEED_WEIGHT
        # print(current_data['speed'])
        if len(prev_data)==0:
            return False
        else:
            return current_data['speed']<=0.5 and prev_data['speed']<=0.5
        
    def isJam(self):
        return False
    
    def isPeople(self):
        return False
