import threading
import queue
import time
import socket
import struct
import numpy as np
import cv2
import signal
import sys
import pandas as pd
import math
from ultralytics import YOLOv10
from ultralytics.solutions import speed_estimation,object_counter

import cv2
import argparse

# 获取目标类别名称
class_names = [
                        "person", "bicycle", "Car", "motorcycle", "airplane", "bus", "train", "truck", 
                        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
                        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
                        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
                        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
                        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
                        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
                        "hair drier", "toothbrush"
                    ]

# 假设类别对应的物理尺寸（单位：米）
object_sizes = {
    "person": (0.5, 1.7),  # 例如，人的尺寸可以是 宽0.5米，高1.7米
    "bicycle": (0.6, 1.8),  # 自行车的尺寸：宽0.6米，长1.8米
    "car": (1.8, 5.9),  # 汽车的尺寸：宽1.8米，长4.5米
    "motorcycle": (0.8, 1.8),  # 摩托车的尺寸：宽0.8米，长1.8米
    "airplane": (20.0, 30.0),  # 飞机的尺寸：宽20米，长30米
    "bus": (2.5, 12.0),  # 公交车的尺寸：宽2.5米，长12米
    "train": (3.0, 25.0),  # 火车的尺寸：宽3米，长25米
    "truck": (2.5, 10.0),  # 卡车的尺寸：宽2.5米，长6米
    "boat": (1.5, 10.0),  # 船的尺寸：宽1.5米，长4米
    "traffic light": (0.3, 0.3),  # 交通灯的尺寸：宽0.3米，高0.3米
    "fire hydrant": (0.3, 0.3),  # 消防栓的尺寸：宽0.3米，高0.3米
    "stop sign": (0.6, 0.6),  # 停止标志的尺寸：宽0.6米，长0.6米
    "parking meter": (0.3, 0.3),  # 停车计时器的尺寸：宽0.3米，长0.3米
    "bench": (0.5, 1.5),  # 长椅的尺寸：宽0.5米，长1.5米
    "bird": (0.2, 0.3),  # 鸟的尺寸：宽0.2米，长0.3米
    "cat": (0.3, 0.4),  # 猫的尺寸：宽0.3米，长0.4米
    "dog": (0.4, 0.5),  # 狗的尺寸：宽0.4米，长0.5米
    "horse": (1.0, 2.0),  # 马的尺寸：宽1.0米，长2.0米
    "sheep": (0.7, 1.2),  # 羊的尺寸：宽0.7米，长1.2米
    "cow": (1.0, 2.5),  # 牛的尺寸：宽1.0米，长2.5米
    "elephant": (2.0, 4.0),  # 大象的尺寸：宽2.0米，长4.0米
    "bear": (1.0, 2.0),  # 熊的尺寸：宽1.0米，长2.0米
    "zebra": (1.0, 2.0),  # 斑马的尺寸：宽1.0米，长2.0米
    "giraffe": (1.5, 4.0),  # 长颈鹿的尺寸：宽1.5米，长4.0米
    "backpack": (0.2, 0.3),  # 背包的尺寸：宽0.2米，长0.3米
    "umbrella": (0.5, 1.0),  # 雨伞的尺寸：宽0.5米，长1.0米
    "handbag": (0.2, 0.3),  # 手袋的尺寸：宽0.2米，长0.3米
    "tie": (0.1, 0.2),  # 领带的尺寸：宽0.1米，长0.2米
    "suitcase": (0.5, 0.6),  # 行李箱的尺寸：宽0.5米，长0.6米
    "frisbee": (0.2, 0.2),  # 飞盘的尺寸：宽0.2米，长0.2米
    "skis": (0.2, 1.5),  # 滑雪板的尺寸：宽0.2米，长1.5米
    "snowboard": (0.2, 1.8),  # 滑雪板的尺寸：宽0.2米，长1.8米
    "sports ball": (0.2, 0.2),  # 体育球的尺寸：宽0.2米，长0.2米
    "kite": (1.0, 1.0),  # 风筝的尺寸：宽1.0米，长1.0米
    "baseball bat": (0.1, 1.0),  # 棒球棒的尺寸：宽0.1米，长1.0米
    "baseball glove": (0.2, 0.3),  # 棒球手套的尺寸：宽0.2米，长0.3米
    "skateboard": (0.5, 1.0),  # 滑板的尺寸：宽0.5米，长1.0米
    "surfboard": (0.5, 2.0),  # 冲浪板的尺寸：宽0.5米，长2.0米
    "tennis racket": (0.3, 0.6),  # 网球拍的尺寸：宽0.3米，长0.6米
    "bottle": (0.1, 0.3),  # 瓶子的尺寸：宽0.1米，长0.3米
    "wine glass": (0.1, 0.3),  # 酒杯的尺寸：宽0.1米，长0.3米
    "cup": (0.1, 0.1),  # 杯子的尺寸：宽0.1米，长0.1米
    "fork": (0.02, 0.2),  # 叉子的尺寸：宽0.02米，长0.2米
    "knife": (0.02, 0.2),  # 刀子的尺寸：宽0.02米，长0.2米
    "spoon": (0.02, 0.2),  # 勺子的尺寸：宽0.02米，长0.2米
    "bowl": (0.2, 0.2),  # 碗的尺寸：宽0.2米，长0.2米
    "banana": (0.1, 0.2),  # 香蕉的尺寸：宽0.1米，长0.2米
    "apple": (0.1, 0.1),  # 苹果的尺寸：宽0.1米，长0.1米
    "sandwich": (0.2, 0.2),  # 三明治的尺寸：宽0.2米，长0.2米
    "orange": (0.1, 0.1),  # 橙子的尺寸：宽0.1米，长0.1米
    "broccoli": (0.2, 0.2),  # 西蓝花的尺寸：宽0.2米，长0.2米
    # 继续为其它类别添加尺寸
}

def get_object_size(class_name):
    # 如果类别在字典中，返回对应的尺寸；否则返回默认值
    return object_sizes.get(class_name, (0, 0))  # 如果类别不存在，返回默认值(0, 0)
# 初始化YOLOv10模型
yolov10_model = YOLOv10("fake-yolo.pt")
names = yolov10_model.model.names
# 设置测速线段的两个端点,一条直线,(x,y)
line_pts = [(0, 615), (1920, 615)]
# 初始化速度估计器
speed_obj = speed_estimation.SpeedEstimator()
# 设置速度估计器的参数，包括测速线段、对象名称和是否实时显示图像
# 计数区域或线。只有出现在指定区域内或穿过指定线的对象才会被计数。
speed_obj.set_args(reg_pts=line_pts, names=names, view_img=False)

# 初始化计数器
counter_obj = object_counter.ObjectCounter()
counter_obj.set_args(reg_pts=line_pts, classes_names=names, view_img=False)

# 公式参数
params_width = [-30843.21, 300.83]
params_height = [-29908.30, 284.79]

width_coeff = [ 0.44001963 ,-5.30381413]  # 线性拟合（一次多项式）
# 对于高度
height_coeff = [ 0.29965956,20.64846326]

# 根据拟合结果构造函数，输入y值，输出框的宽度和高度
def calculate_box_size(y):
    width = np.polyval(width_coeff, y)  # 计算宽度
    height = np.polyval(height_coeff, y)  # 计算高度
    return int(width), int(height)


frame_width = 1920
frame_height = 1080
# 创建 VideoWriter 对象
output_filename = "output.mp4"
video_writer = cv2.VideoWriter(
    output_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),  # 编码器
    30,  # 帧率
    (frame_width, frame_height)  # 视频帧大小
)

def count_non_motorized_vehicles(df):
    """
    计算检测时刻区域内非机动车的数量。

    :param df: DataFrame，包含车辆的id, Time, type, x, y等信息
    :return: 非机动车数量
    """
    # 假设type=1代表非机动车
    non_motorized_vehicles = df[df['type'] == 1]

    # 统计非机动车数量（基于id去重）
    non_motorized_count = non_motorized_vehicles['id'].nunique()

    return non_motorized_count


def calculate_non_motorized_occupancy(df, x1, x2, y1, y2):
    """
    计算非机动车占有率。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 非机动车占有率
    """
    # 假设type=1表示非机动车
    non_motorized_vehicles = df[df['type'] == 1]

    # 计算区域的总面积
    total_area = (x2 - x1) * (y2 - y1)

    # 计算非机动车占有的总面积
    total_non_motorized_area = non_motorized_vehicles['length'] * non_motorized_vehicles['width']
    total_non_motorized_area = total_non_motorized_area.sum()  # 累加所有非机动车的占地面积

    # 计算非机动车占有率
    occupancy_rate = total_non_motorized_area / total_area

    return occupancy_rate


def calculate_average_non_motorized_occupancy(df, x1, x2, y1, y2):
    """
    按时间分组计算非机动车占有率的算术平均值。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 非机动车占有率的算术平均值
    """
    # 按时间分组
    time_groups = df.groupby('Time')

    occupancy_rates = []

    # 对每个时间点计算占有率
    for _, group in time_groups:
        rate = calculate_non_motorized_occupancy(group, x1, x2, y1, y2)
        occupancy_rates.append(rate)

    # 计算所有时间点占有率的平均值
    if occupancy_rates:
        average_occupancy = sum(occupancy_rates) / len(occupancy_rates)
    else:
        average_occupancy = 0

    return average_occupancy

def create_area_from_points(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    根据四个点的坐标创建一个行人检测区域，并以列表形式返回该区域坐标。

    :param x1, y1, x2, y2, x3, y3, x4, y4: 四个点的坐标，定义一个矩形或任意四边形区域
    :return: 一个列表，包含四个点的坐标（[x1, y1, x2, y2, x3, y3, x4, y4]）
    """
    # 将四个点的坐标按顺序存储为列表
    area = [x1, y1, x2, y2, x3, y3, x4, y4]

    return area


def count_pedestrians_in_area(df, x1, x2, y1, y2):
    """
    计算检测时刻区域内行人的数量
    既可用于实时数据，也可以用于统计数据
    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 在该区域内的行人数量
    """
    # 过滤出行人数据（假设行人的type = 0）
    pedestrians = df[df['type'] == 0]

    # 过滤出在指定区域内的行人
    pedestrians_in_area = pedestrians[(pedestrians['x'] >= x1) & (pedestrians['x'] <= x2) &
                                      (pedestrians['y'] >= y1) & (pedestrians['y'] <= y2)]

    # 统计在该区域内的行人数量（基于id去重）
    pedestrian_count = pedestrians_in_area['id'].nunique()

    return pedestrian_count


def calculate_area_occupancy(df, x1, x2, y1, y2):
    """
    计算检测时刻区域内行人所占面积占区域总面积的比例

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 行人所占面积占区域总面积的比例
    """
    # 过滤出行人数据（假设行人的type = 0）
    pedestrians = df[df['type'] == 0]

    # 过滤出在指定区域内的行人
    pedestrians_in_area = pedestrians[(pedestrians['x'] >= x1) & (pedestrians['x'] <= x2) &
                                      (pedestrians['y'] >= y1) & (pedestrians['y'] <= y2)]

    # 计算总区域面积
    total_area = (x2 - x1) * (y2 - y1)

    # 计算所有行人所占的总面积，假设每个行人占据的面积是 length * width
    # 如果没有宽度信息，可以假设宽度为某个固定值
    pedestrians_area = pedestrians_in_area['length'] * pedestrians_in_area['width']

    # 计算行人占用的总面积
    total_pedestrian_area = pedestrians_area.sum()

    # 计算比例
    occupancy_ratio = total_pedestrian_area / total_area if total_area > 0 else 0

    return occupancy_ratio


def calculate_average_occupancy_by_time(df, x1, x2, y1, y2):
    """
    按检测时刻分组计算行人占有率的平均值。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 行人占有率的算数平均值
    """
    # 按时间分组
    time_groups = df.groupby('Time')

    occupancy_ratios = []

    # 对每个时间点计算行人占有率
    for _, group in time_groups:
        ratio = calculate_area_occupancy(group, x1, x2, y1, y2)
        occupancy_ratios.append(ratio)

    # 计算所有时间点占有率的平均值
    if occupancy_ratios:
        overall_average_occupancy = sum(occupancy_ratios) / len(occupancy_ratios)
    else:
        overall_average_occupancy = 0

    return overall_average_occupancy


class PedestrianTracking:
    def __init__(self, detection_interval):
        self.detection_interval = detection_interval  # 单位检测间隔（秒）
        self.pedestrian_data = {}  # 用于跟踪每个行人的进入时间与累计等待时间

    def update_waiting_time(self, df, x1, x2, y1, y2):
        """
        更新检测时刻等待区内所有行人的平均等待时间。

        :param df: DataFrame，包含行人数据（id, Time, x, y等信息）
        :param x1, x2, y1, y2: 等待区区域的四个边界坐标
        :return: 当前时刻等待区内所有行人的平均等待时间
        """
        # 获取当前区域内的行人
        pedestrians_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

        # 更新每个行人的等待时间
        updated_waiting_time = {}
        total_waiting_time = 0  # 用于计算平均等待时间
        total_pedestrians = 0  # 统计当前等待区内行人数量

        for _, row in pedestrians_in_area.iterrows():
            pedestrian_id = row['id']

            # 如果该行人已经存在于tracking中，累加等待时间
            if pedestrian_id in self.pedestrian_data:
                self.pedestrian_data[pedestrian_id] += self.detection_interval
            else:
                self.pedestrian_data[pedestrian_id] = self.detection_interval  # 初始化等待时间

            updated_waiting_time[pedestrian_id] = self.pedestrian_data[pedestrian_id]
            total_waiting_time += self.pedestrian_data[pedestrian_id]
            total_pedestrians += 1

        # 如果没有行人在等待区，则返回平均等待时间为0
        if total_pedestrians == 0:
            return 0

        # 计算当前时刻的平均等待时间
        average_waiting_time = total_waiting_time / total_pedestrians
        return average_waiting_time


def count_upward_pedestrians_on_zebra_crossing(df, x1, x2, y1, y2):
    """
    计算检测时刻斑马线上上行方向的行人数量。
    既可用于检测数据，也可以用于统计数据
    :param df: DataFrame，包含行人数据（id, Time, x, y, yaw等信息）
    :param x1, x2, y1, y2: 斑马线的区域边界坐标
    :return: 在斑马线上上行方向的行人数量
    """
    # 筛选出在斑马线区域内的行人
    pedestrians_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 计算行人的上行方向，假设yaw在0到180度之间为上行,根据实际情况调整
    upward_pedestrians = pedestrians_in_area[pedestrians_in_area['yaw'] >= 0]
    upward_pedestrians = upward_pedestrians[upward_pedestrians['yaw'] <= 180]

    # 统计上行方向的行人数量
    upward_count = upward_pedestrians['id'].nunique()

    return upward_count


def count_pedestrians_on_zebra_crossing(df, x1, x2, y1, y2):
    """
    计算检测时刻斑马线上下行方向的行人数量。
    既可用于检测数据，也可以用于统计数据
    :param df: DataFrame，包含行人数据（id, Time, x, y, yaw等信息）
    :param x1, x2, y1, y2: 斑马线的区域边界坐标
    :return: 上行方向和下行方向的行人数量
    """
    # 筛选出在斑马线区域内的行人
    pedestrians_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 下行方向：yaw在180到360度之间，可依据实时数据改变条件
    downward_pedestrians = pedestrians_in_area[pedestrians_in_area['yaw'] > 180]
    downward_pedestrians = downward_pedestrians[downward_pedestrians['yaw'] <= 360]

    downward_count = downward_pedestrians['id'].nunique()

    return downward_count

class PedestrianTracking:
    def __init__(self, detection_interval):
        self.detection_interval = detection_interval  # 单位检测间隔（秒）
        self.pedestrian_data = {}  # 用于跟踪每个行人的进入时间与累计等待时间
        self.first_pedestrian_enter_time = None  # 记录首个进入等待区的行人

    def update_waiting_time(self, df, x1, x2, y1, y2):
        """
        更新检测时刻等待区内首个进入等待区的行人的等待时间。

        :param df: DataFrame，包含行人数据（id, Time, x, y等信息）
        :param x1, x2, y1, y2: 等待区区域的四个边界坐标
        :return: 当前时刻等待区内首个进入等待区的行人的等待时间
        """
        # 获取当前区域内的行人
        pedestrians_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

        # 记录当前时刻首个进入等待区的行人的等待时间
        first_pedestrian_waiting_time = None

        for _, row in pedestrians_in_area.iterrows():
            pedestrian_id = row['id']
            current_time = row['Time']

            # 如果首个进入等待区的行人还没有记录，则记录
            if self.first_pedestrian_enter_time is None:
                self.first_pedestrian_enter_time = pedestrian_id
                self.pedestrian_data[pedestrian_id] = 0  # 初始化为0秒

            # 如果是首个进入的行人，则累加等待时间
            if pedestrian_id == self.first_pedestrian_enter_time:
                self.pedestrian_data[pedestrian_id] += self.detection_interval
                first_pedestrian_waiting_time = self.pedestrian_data[pedestrian_id]

        # 如果没有行人在等待区，则返回等待时间为0
        if first_pedestrian_waiting_time is None:
            return 0

        return first_pedestrian_waiting_time

def count_vehicles_by_class(vehicle_data):
    """
    统计各类车辆的数量（A类：车长≥12m，B类：6m≤车长<12m，C类：车长<6m）。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆数据，包括以下列：
                         - 'id': 目标的ID
                         - 'type': 目标类型，0代表行人，其它代表不同类型的车辆
                         - 'x': 目标的x坐标
                         - 'y': 目标的y坐标
                         - 'xv': x方向速度分量
                         - 'yv': y方向速度分量
                         - 'yaw': 目标的航向角
                         - 'length': 目标的长度（单位：米）
    :return: A类车、B类车和C类车的数量
    """
    # 去重，只保留每个车辆的第一条数据（通过'id'字段）
    vehicle_data_unique = vehicle_data.drop_duplicates(subset='id')

    # 定义A类、B类、C类车辆数量
    Acount = 0
    Bcount = 0
    Ccount = 0

    # 遍历车辆数据，计算每个车辆的类别并进行统计
    for _, vehicle in vehicle_data_unique.iterrows():
        # 目标是车辆（排除行人,此处假设行人的type值为0）
        if vehicle['type'] != 0:
            # 判断车辆长度，进行分类统计
            if vehicle['length'] >= 12:
                Acount += 1
            elif 6 <= vehicle['length'] < 12:
                Bcount += 1
            elif vehicle['length'] < 6:
                Ccount += 1

    return Acount, Bcount, Ccount


def calculate_occupancy_rate(vehicle_data, detection_interval):
    """
    计算车辆占用率，基于输入的DataFrame与检测间隔。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆数据，包括以下列：
                         - 'id': 车辆的ID
                         - 'Time': 车辆占用检测器的时间戳（单位：秒）
                         - 'type': 车辆类型，0代表行人，其他为车辆类型
                         - 'x': 车辆的x坐标
                         - 'y': 车辆的y坐标
                         - 'xv': x方向速度分量
                         - 'yv': y方向速度分量
                         - 'yaw': 车辆的航向角
                         - 'length': 车辆的长度（单位：米）
    :return: 时间占有率（0 到 1 之间的比例）
    :detection_interval:检测器的检测间隔或统计周期，依要求输入（单位：秒）
    """

    # 记录每辆车的占用时间段
    vehicle_time_segments = []

    # 遍历车辆数据，计算占用时间段
    for _, vehicle in vehicle_data.iterrows():
        if vehicle['type'] == 0:  # 排除行人
            continue

        start_time = vehicle['Time']
        end_time = start_time + 1  # 假设每辆车占用的时间是从Time到Time+1秒

        vehicle_time_segments.append((start_time, end_time))

    # 按时间排序
    vehicle_time_segments.sort()

    # 合并重叠的时间段
    merged_time_segments = []
    if vehicle_time_segments:
        current_start, current_end = vehicle_time_segments[0]

        for start, end in vehicle_time_segments[1:]:
            if start <= current_end:  # 时间段重叠或相接
                current_end = max(current_end, end)  # 合并时间段
            else:
                merged_time_segments.append((current_start, current_end))  # 记录上一个时间段
                current_start, current_end = start, end  # 更新为新的时间段

        # 添加最后一个时间段
        merged_time_segments.append((current_start, current_end))

    # 计算所有合并后的时间段的总占用时间
    total_occupancy_time = 0
    for start, end in merged_time_segments:
        total_occupancy_time += (end - start)

    # 计算时间占有率
    occupancy_rate = total_occupancy_time / detection_interval
    return min(occupancy_rate, 1.0)  # 确保占用率不超过1


def calculate_average_speed(vehicle_data):
    """
    计算在给定检测间隔内通过车辆的平均速度。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆数据，包括以下列：
                         - 'id': 车辆的ID
                         - 'Time': 车辆的时间戳（单位：秒）
                         - 'type': 车辆类型，0代表行人，其他为车辆类型
                         - 'x': 车辆的x坐标
                         - 'y': 车辆的y坐标
                         - 'xv': x方向速度分量
                         - 'yv': y方向速度分量
                         - 'yaw': 车辆的航向角
                         - 'length': 车辆的长度（单位：米）
    :return: 检测间隔内通过车辆的平均速度（单位：米/秒）
    :detection_interval:检测器的检测间隔（单位：秒）
    """

    total_speed = 0  # 累计车辆的总速度
    count = 0  # 计算通过车辆的数量

    # 过滤出类型为车辆的数据（不包含行人）
    vehicles = vehicle_data[vehicle_data['type'] != 0]

    # 计算每辆车的速度并进行累加
    for _, vehicle in vehicles.iterrows():
        # 计算车辆的速度
        speed = math.sqrt(vehicle['xv'] ** 2 + vehicle['yv'] ** 2)
        total_speed += speed
        count += 1

    # 计算平均速度
    if count == 0:
        return 0  # 如果没有车辆通过，则平均速度为0
    else:
        average_speed = total_speed / count
        return average_speed


def calculate_average_length(vehicle_data):
    """
    计算在给定检测间隔内通过车辆的平均长度。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆数据，包括以下列：
                         - 'id': 车辆的ID
                         - 'Time': 车辆的时间戳（单位：秒）
                         - 'type': 车辆类型，0代表行人，其他为车辆类型
                         - 'x': 车辆的x坐标
                         - 'y': 车辆的y坐标
                         - 'xv': x方向速度分量
                         - 'yv': y方向速度分量
                         - 'yaw': 车辆的航向角
                         - 'length': 车辆的长度（单位：米）
    :return: 检测间隔内通过车辆的平均长度（单位：米）
    """

    total_length = 0  # 累计车辆的总长度
    count = 0  # 计算通过车辆的数量

    # 过滤出类型为车辆的数据（不包含行人）
    vehicles = vehicle_data[vehicle_data['type'] != 0]

    # 累计车辆的长度
    for _, vehicle in vehicles.iterrows():
        total_length += vehicle['length']
        count += 1

    # 计算平均长度
    if count == 0:
        return 0  # 如果没有车辆通过，则平均长度为0
    else:
        average_length = total_length / count
        return average_length

def calculate_average_time_gap(vehicle_data):
    """
    计算在给定检测间隔内，车辆与上一辆车的平均时间间隔（以车头为基准）。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆数据，包括以下列：
                         - 'id': 车辆的ID
                         - 'Time': 车辆车头进入检测器的时间（单位：秒）
                         - 'type': 车辆类型，0代表行人，其他为车辆类型
                         - 'x': 车辆的x坐标
                         - 'y': 车辆的y坐标
                         - 'xv': x方向速度分量
                         - 'yv': y方向速度分量
                         - 'yaw': 车辆的航向角
                         - 'length': 车辆的长度（单位：米）
    :return: 平均时间间隔（单位：秒），若只有一辆车或没有车辆，则返回0。
    """
    # 过滤出类型为车辆的行（排除行人，即type != 0）
    vehicles = vehicle_data[vehicle_data['type'] != 0]

    # 如果没有车辆或只有一辆车，返回0
    if len(vehicles) <= 1:
        return 0

    # 按照时间顺序排序
    vehicles_sorted = vehicles.sort_values(by='Time')

    # 计算相邻车辆之间的时间间隔
    time_gaps = vehicles_sorted['Time'].diff().dropna()  # 计算时间差，去掉NaN值

    # 计算平均时间间隔
    average_time_gap = time_gaps.mean()

    return average_time_gap


def calculate_average_parking_count(vehicle_data, speed_threshold=0.1):
    """
    计算在给定检测间隔内通过车辆的平均停车次数。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆数据，包括以下列：
                         - 'id': 车辆的ID
                         - 'Time': 车辆车头进入检测器的时间（单位：秒）
                         - 'type': 车辆类型，0代表行人，其他为车辆类型
                         - 'x': 车辆的x坐标
                         - 'y': 车辆的y坐标
                         - 'xv': x方向速度分量
                         - 'yv': y方向速度分量
                         - 'yaw': 车辆的航向角
                         - 'length': 车辆的长度（单位：米）
    :param speed_threshold: 速度阈值，小于该值视为停车，单位：米/秒
    :return: 平均停车次数
    """
    total_parking_count = 0  # 累计停车次数
    count = 0  # 计算通过的车辆数量

    # 过滤掉行人，只处理类型为车辆的数据
    vehicles = vehicle_data[vehicle_data['type'] != 0]

    # 按id分组
    grouped = vehicles.groupby('id')

    # 遍历每辆车
    for vehicle_id, group in grouped:
        # 按时间排序
        group = group.sort_values(by='Time')

        # 初始化停车状态
        is_parking = False
        parking_count = 0

        # 遍历该车辆的所有记录，检查其速度,车辆速度连续低于速度阈值仅视为一次停车
        for _, row in group.iterrows():
            speed = math.sqrt(row['xv'] ** 2 + row['yv'] ** 2)

            if speed < speed_threshold:
                if not is_parking:
                    is_parking = True
                    parking_count += 1  # 进入停车状态，计数一次停车
            else:
                is_parking = False  # 车辆停止停车

        # 累计停车次数
        total_parking_count += parking_count
        count += 1  # 统计车辆数量

    # 计算平均停车次数
    if count == 0:
        return 0  # 如果没有车辆通过，则平均停车次数为0
    else:
        average_parking_count = total_parking_count / count
        return average_parking_count




def calculate_average_parking_duration(vehicle_data, parking_speed_threshold=0.1):
    """
    计算检测间隔内所有车辆的平均停车时长。
    本代码既可用于检测数据，也可用于统计数据
    :param vehicle_data: DataFrame，包含车辆的数据，每行代表一辆车的记录。
    :param parking_speed_threshold: 速度阈值，低于该阈值判定为停车状态
    :return: 平均停车时长，单位为秒
    """
    # 存储所有车辆的停车时长
    parking_durations = []

    # 按id分组，处理每辆车的数据
    for vehicle_id, group in vehicle_data.groupby('id'):
        # 按时间排序，保证车辆运动状态的顺序
        group = group.sort_values(by='Time')

        # 计算停车时长
        parking_time = 0
        is_parking = False
        for i, row in group.iterrows():
            # 计算车辆的总速度（xv 和 yv）
            total_speed = np.sqrt(row['xv'] ** 2 + row['yv'] ** 2)

            if total_speed < parking_speed_threshold:  # 判断停车状态
                if not is_parking:
                    is_parking = True
                    parking_start_time = row['Time']
            else:
                if is_parking:
                    is_parking = False
                    parking_time += row['Time'] - parking_start_time

        # 如果最后一个记录还是停车状态，累加停车时间
        if is_parking:
            parking_time += group.iloc[-1]['Time'] - parking_start_time

        parking_durations.append(parking_time)

    # 计算平均停车时长
    average_parking_duration = np.mean(parking_durations) if parking_durations else 0

    return average_parking_duration

def calculate_sampling_count(sampling_frequency, sampling_interval):
    """
    根据采样频率和检测间隔计算系统的采样次数。

    :param sampling_frequency: 系统的采样频率 (Hz)，即每秒采样多少次
    :param sampling_interval: 当前检测间隔，单位秒
    :return: 当前检测间隔的采样次数
    """
    sampling_count = sampling_frequency * sampling_interval
    return int(sampling_count)


def generate_occupancy_sequence(vehicle_data, sampling_frequency, area_bounds):
    """
    生成断面或线圈被车辆占有的状态序列。

    :param vehicle_data: DataFrame，车辆数据，每行包含车辆的信息
    :param sampling_frequency: 系统采样频率，单位Hz
    :param area_bounds: 断面或线圈的区域边界，假设为矩形区域 (xmin, xmax, ymin, ymax)
    :return: 车辆在检测间隔内的占有状态序列，1表示占有，0表示不占有
    """
    # 定义状态序列
    occupancy_sequence = []

    # 计算每个采样点的时间间隔
    time_interval = 1 / sampling_frequency  # 每个采样点的时间间隔（秒）

    # 遍历每一辆车
    for _, vehicle in vehicle_data.iterrows():
        vehicle_id = vehicle['id']
        vehicle_time = vehicle['Time']
        vehicle_x = vehicle['x']
        vehicle_y = vehicle['y']
        vehicle_length = vehicle['length']

        # 判断车辆是否占有区域的条件
        # 假设车辆的车头位置(x, y)，车长为length，判断车辆头部到尾部的区域是否覆盖断面或线圈
        vehicle_end_x = vehicle_x + vehicle_length * np.cos(np.radians(vehicle['yaw']))
        vehicle_end_y = vehicle_y + vehicle_length * np.sin(np.radians(vehicle['yaw']))

        # 判断车辆是否处于断面/线圈区域内
        is_in_area = (min(vehicle_x, vehicle_end_x) <= area_bounds[1] and
                      max(vehicle_x, vehicle_end_x) >= area_bounds[0] and
                      min(vehicle_y, vehicle_end_y) <= area_bounds[3] and
                      max(vehicle_y, vehicle_end_y) >= area_bounds[2])

        # 对于该车，如果在每个采样点占有区域，状态为1，否则为0
        num_samples = int(sampling_frequency * vehicle_time)  # 计算采样点的数量
        if is_in_area:
            occupancy_sequence.extend([1] * num_samples)
        else:
            occupancy_sequence.extend([0] * num_samples)

    return occupancy_sequence

def count_a_class_vehicles_in_area(df, x1, x2, y1, y2):
    """
    计算在指定区域内通过的A类车辆数量（车长大于等于12米）

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length等信息
    :param x1, x2, y1, y2: 检测区域的四个边界坐标
    :return: 在该区域内的A类车辆数量
    """
    # 过滤出A类车辆（车长大于12m）
    a_class_vehicles = df[df['length'] >= 12]

    # 过滤出在指定区域内的车辆
    vehicles_in_area = a_class_vehicles[(a_class_vehicles['x'] >= x1) & (a_class_vehicles['x'] <= x2) &
                                        (a_class_vehicles['y'] >= y1) & (a_class_vehicles['y'] <= y2)]

    # 统计在该区域内的A类车辆数量（基于id去重）
    a_class_count = vehicles_in_area['id'].nunique()

    return a_class_count

def count_b_class_vehicles_in_area(df, x1, x2, y1, y2):
    """
    计算在指定区域内通过的B类车辆数量（车长大于等于6米，小于12米）

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 在该区域内的B类车辆数量
    """
    # 过滤出B类车辆（车长大于等于6m且小于12m）
    b_class_vehicles = df[(df['length'] >= 6) & (df['length'] < 12)]

    # 过滤出在指定区域内的车辆
    vehicles_in_area = b_class_vehicles[(b_class_vehicles['x'] >= x1) & (b_class_vehicles['x'] <= x2) &
                                        (b_class_vehicles['y'] >= y1) & (b_class_vehicles['y'] <= y2)]

    # 统计在该区域内的B类车辆数量（基于id去重）
    b_class_count = vehicles_in_area['id'].nunique()

    return b_class_count

def count_c_class_vehicles_in_area(df, x1, x2, y1, y2):
    """
    计算在指定区域内通过的C类车辆数量（车长小于6米）

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 在该区域内的C类车辆数量
    """
    # 过滤出C类车辆（车长小于6m）
    c_class_vehicles = df[df['length'] < 6]

    # 过滤出在指定区域内的车辆
    vehicles_in_area = c_class_vehicles[(c_class_vehicles['x'] >= x1) & (c_class_vehicles['x'] <= x2) &
                                        (c_class_vehicles['y'] >= y1) & (c_class_vehicles['y'] <= y2)]

    # 统计在该区域内的C类车辆数量（基于id去重）
    c_class_count = vehicles_in_area['id'].nunique()

    return c_class_count


def calculate_occupied_area_ratio(df, x1, x2, y1, y2):
    """
    计算在检测时刻，区域内被车辆占有的面积占区域总面积的比例
    在此过程中，按车辆ID去重，确保每辆车只计算一次占用面积。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 车辆占用面积占区域总面积的比例
    """
    # 计算区域总面积
    total_area = (x2 - x1) * (y2 - y1)

    # 过滤出所有车辆数据，并去重，确保每辆车只计算一次
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 按id去重，确保每辆车只计算一次
    vehicles_in_area_unique = vehicles_in_area.drop_duplicates(subset='id')

    # 计算车辆占用的总面积
    total_occupied_area = 0
    for _, row in vehicles_in_area_unique.iterrows():
        # 每辆车的占用面积：length * width
        total_occupied_area += row['length'] * row['width']

    # 计算被占有面积的比例
    if total_area == 0:
        return 0  # 避免除以0的错误
    else:
        occupied_area_ratio = total_occupied_area / total_area

    return occupied_area_ratio


def calculate_average_occupied_area_ratio(df, x1, x2, y1, y2):
    """
    计算区域在不同检测时刻的占用面积比例，并返回算数平均值。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 区域被车辆占有面积的平均占用比例
    """
    # 按时间分组
    df['Time'] = pd.to_datetime(df['Time'], format='%M:%S.%f').dt.strftime('%M:%S')  # 格式化为时间格式
    time_groups = df.groupby('Time')

    occupied_area_ratios = []

    # 对每个时间点计算占用面积比例
    for _, group in time_groups:
        ratio = calculate_occupied_area_ratio(group, x1, x2, y1, y2)
        occupied_area_ratios.append(ratio)

    # 计算平均占用比例
    if occupied_area_ratios:
        average_ratio = sum(occupied_area_ratios) / len(occupied_area_ratios)
    else:
        average_ratio = 0

    return average_ratio


def calculate_queue_length(df, x0, speed_threshold=0.1):
    """
    计算检测时刻区域内的排队长度（只考虑停车车辆，以停车线所在边或底线起始边为参考基准）

    :param df: DataFrame，包含车辆的id, x, y, length, yaw, xv, yv等信息
    :param x0: 停车线所在边或底线的起始位置（参考基准）
    :param speed_threshold: 判断车辆是否停车的速度阈值（单位：米/秒）
    :return: 排队长度
    """
    # 判断车辆是否停车：速度小于阈值的车辆视为已停车
    df['is_parked'] = (df['xv'].abs() < speed_threshold) & (df['yv'].abs() < speed_threshold)

    # 只选择已停车的车辆
    parked_vehicles = df[df['is_parked']]

    # 如果没有已停车车辆，返回排队长度为0
    if parked_vehicles.empty:
        return 0

    # 将yaw角度转换为弧度
    parked_vehicles['yaw_rad'] = np.radians(parked_vehicles['yaw'])

    # 计算每辆车的前端位置
    parked_vehicles['front_x'] = parked_vehicles['x'] - (parked_vehicles['length'] / 2) * np.cos(
        parked_vehicles['yaw_rad'])
    parked_vehicles['front_y'] = parked_vehicles['y'] - (parked_vehicles['length'] / 2) * np.sin(
        parked_vehicles['yaw_rad'])

    # 排序停车车辆，按照前端位置x从小到大排序
    parked_vehicles_sorted = parked_vehicles.sort_values(by='front_x')

    # 最远停车车辆的前端位置
    furthest_vehicle_front_x = parked_vehicles_sorted.iloc[-1]['front_x']

    # 排队长度
    queue_length = furthest_vehicle_front_x - x0

    return queue_length


def calculate_average_queue_length(df, x0, speed_threshold=0.1):
    """
    计算不同检测时刻的排队长度，并返回排队长度的平均值。

    :param df: DataFrame，包含车辆的id, x, y, length, yaw, xv, yv等信息
    :param x0: 停车线所在边或底线的起始位置（参考基准）
    :param speed_threshold: 判断车辆是否停车的速度阈值（单位：米/秒）
    :return: 排队长度的平均值
    """
    # 按时间分组
    df['Time'] = pd.to_datetime(df['Time'], format='%M:%S.%f').dt.strftime('%M:%S')  # 格式化为时间格式
    time_groups = df.groupby('Time')

    queue_lengths = []

    # 对每个时间点计算排队长度
    for _, group in time_groups:
        length = calculate_queue_length(group, x0, speed_threshold)
        queue_lengths.append(length)

    # 计算平均排队长度
    if queue_lengths:
        average_queue_length = sum(queue_lengths) / len(queue_lengths)
    else:
        average_queue_length = 0

    return average_queue_length


def calculate_first_vehicle_distance_to_parking_line(df, x1, x2, y1, y2, x11, y11, x21, y21):
    """
    计算检测时刻区域内首辆车与停车线的距离。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :param x11, y11, x21, y21: 停车线的两个端点坐标
    :return: 首辆车与停车线的距离（单位：米）
    """
    # 过滤出所有车辆数据，并找到在指定区域内的车辆
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 计算停车线的参数
    A = y21 - y11
    B = x11 - x21
    C = x21 * y11 - x11 * y21

    # 计算每辆车的x, y坐标到停车线的距离
    vehicles_in_area['distance_to_parking_line'] = abs(
        A * vehicles_in_area['x'] + B * vehicles_in_area['y'] + C) / np.sqrt(A ** 2 + B ** 2)

    # 找到距离停车线最近的停车车辆，即首辆车
    first_vehicle = vehicles_in_area.loc[vehicles_in_area['distance_to_parking_line'].idxmin()]

    return first_vehicle['distance_to_parking_line']


def calculate_last_vehicle_distance_to_parking_line(df, x1, x2, y1, y2, x11, y11, x21, y21):
    """
    计算检测时刻区域内末辆车与停车线起始边的距离。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :param x11, y11, x21, y21: 停车线的两个端点坐标
    :return: 末辆车与停车线起始边的距离（单位：米）
    """
    # 过滤出所有车辆数据，并找到在指定区域内的车辆
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 计算停车线的参数
    A = y21 - y11
    B = x11 - x21
    C = x21 * y11 - x11 * y21

    # 计算每辆车的x, y坐标到停车线的距离
    vehicles_in_area['distance_to_parking_line'] = abs(
        A * vehicles_in_area['x'] + B * vehicles_in_area['y'] + C) / np.sqrt(A ** 2 + B ** 2)

    # 找到距离停车线最远的车辆，即末辆车
    last_vehicle = vehicles_in_area.loc[vehicles_in_area['distance_to_parking_line'].idxmax()]

    return last_vehicle['distance_to_parking_line']


def calculate_average_speed_in_area(df, x1, x2, y1, y2):
    """
    计算检测时刻区域内所有车辆的地点车速平均值。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 区域内所有车辆的车速平均值（单位：米/秒）
    """
    # 过滤出所有车辆数据，并找到在指定区域内的车辆
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    if vehicles_in_area.empty:
        return 0  # 如果区域内没有车辆，返回车速为0

    # 计算每辆车的车速
    vehicles_in_area['speed'] = np.sqrt(vehicles_in_area['xv'] ** 2 + vehicles_in_area['yv'] ** 2)

    # 计算车速的平均值
    average_speed = vehicles_in_area['speed'].mean()

    return average_speed


# def calculate_average_speed(df, x1, x2, y1, y2):
#     """
#     按检测时刻时间分组计算区域内车辆的车速，并返回车速的平均值。

#     :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
#     :param x1, x2, y1, y2: 区域的四个边界坐标
#     :return: 区域内车辆车速的平均值
#     """
#     # 按时间分组
#     df['Time'] = pd.to_datetime(df['Time'], format='%M:%S.%f').dt.strftime('%M:%S')  # 格式化为时间格式
#     time_groups = df.groupby('Time')

#     average_speeds = []

#     # 对每个时间点计算车速平均值
#     for _, group in time_groups:
#         speed = calculate_average_speed_in_area(group, x1, x2, y1, y2)
#         average_speeds.append(speed)

#     # 计算所有时间点车速的平均值
#     if average_speeds:
#         overall_average_speed = sum(average_speeds) / len(average_speeds)
#     else:
#         overall_average_speed = 0

#     return overall_average_speed


def calculate_first_vehicle_info(df, x1, x2, y1, y2, x11, y11, x21, y21):
    """
    计算检测时刻区域内首辆车与停车线的距离以及车速。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :param x11, y11, x21, y21: 停车线的两个端点坐标
    :return: (首辆车与停车线的距离, 首辆车的车速)
    """
    # 过滤出所有车辆数据，并找到在指定区域内的车辆
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 计算停车线的参数
    A = y21 - y11
    B = x11 - x21
    C = x21 * y11 - x11 * y21

    # 计算每辆车的x, y坐标到停车线的距离
    vehicles_in_area['distance_to_parking_line'] = abs(
        A * vehicles_in_area['x'] + B * vehicles_in_area['y'] + C) / np.sqrt(A ** 2 + B ** 2)

    # 找到距离停车线最近的停车车辆，即首辆车
    first_vehicle = vehicles_in_area.loc[vehicles_in_area['distance_to_parking_line'].idxmin()]

    # 计算首辆车的车速
    speed = np.sqrt(first_vehicle['xv'] ** 2 + first_vehicle['yv'] ** 2)

    return first_vehicle['distance_to_parking_line'], speed


def calculate_last_vehicle_info(df, x1, x2, y1, y2, x11, y11, x21, y21):
    """
    计算检测时刻区域内末辆车与停车线的距离以及车速。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :param x11, y11, x21, y21: 停车线的两个端点坐标
    :return: (末辆车与停车线的距离, 末辆车的车速)
    """
    # 过滤出所有车辆数据，并找到在指定区域内的车辆
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 计算停车线的参数
    A = y21 - y11
    B = x11 - x21
    C = x21 * y11 - x11 * y21

    # 计算每辆车的x, y坐标到停车线的距离
    vehicles_in_area['distance_to_parking_line'] = abs(
        A * vehicles_in_area['x'] + B * vehicles_in_area['y'] + C) / np.sqrt(A ** 2 + B ** 2)

    # 找到距离停车线最远的停车车辆，即末辆车
    last_vehicle = vehicles_in_area.loc[vehicles_in_area['distance_to_parking_line'].idxmax()]

    # 计算末辆车的车速
    speed = np.sqrt(last_vehicle['xv'] ** 2 + last_vehicle['yv'] ** 2)

    return last_vehicle['distance_to_parking_line'], speed


def calculate_average_inter_vehicle_distance(df, x1, x2, y1, y2):
    """
    计算检测时刻区域内所有车辆的车间间距的平均值（考虑车辆朝向）。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 车间间距的平均值（单位：米）
    """
    # 过滤出所有车辆数据，并找到在指定区域内的车辆
    vehicles_in_area = df[(df['x'] >= x1) & (df['x'] <= x2) & (df['y'] >= y1) & (df['y'] <= y2)]

    # 计算车辆前端的坐标：前端坐标 = 车辆位置 + (车辆长度 / 2) * cos(yaw), sin(yaw)
    vehicles_in_area['front_x'] = vehicles_in_area['x'] + (vehicles_in_area['length'] / 2) * np.cos(
        np.radians(vehicles_in_area['yaw']))
    vehicles_in_area['front_y'] = vehicles_in_area['y'] + (vehicles_in_area['length'] / 2) * np.sin(
        np.radians(vehicles_in_area['yaw']))

    # 按照前端的x坐标排序，以确保车辆按位置排列
    vehicles_in_area = vehicles_in_area.sort_values(by='front_x')

    # 计算相邻车辆之间的车间间距（欧氏距离）
    vehicle_distances = []
    for i in range(1, len(vehicles_in_area)):
        front_x_prev = vehicles_in_area.iloc[i - 1]['front_x']
        front_y_prev = vehicles_in_area.iloc[i - 1]['front_y']
        front_x_curr = vehicles_in_area.iloc[i]['front_x']
        front_y_curr = vehicles_in_area.iloc[i]['front_y']

        # 计算前端之间的欧氏距离
        distance = np.sqrt((front_x_curr - front_x_prev) ** 2 + (front_y_curr - front_y_prev) ** 2)
        vehicle_distances.append(distance)

    # 如果车辆数少于两个，返回0（无法计算车间间距）
    if len(vehicle_distances) == 0:
        return 0.0

    # 计算车间间距的平均值
    average_distance = np.mean(vehicle_distances)

    return average_distance


def calculate_average_distance_by_time(df, x1, x2, y1, y2):
    """
    按检测时刻时间分组计算车间间距的平均值。

    :param df: DataFrame，包含车辆的id, Time, type, x, y, xv, yv, yaw, length, width等信息
    :param x1, x2, y1, y2: 区域的四个边界坐标
    :return: 所有时间点的车间间距平均值
    """
    # 按时间分组
    time_groups = df.groupby('Time')

    average_distances = []

    # 对每个时间点计算车间间距平均值
    for _, group in time_groups:
        distance = calculate_average_inter_vehicle_distance(group, x1, x2, y1, y2)
        average_distances.append(distance)

    # 计算所有时间点车间间距的平均值
    if average_distances:
        overall_average_distance = np.mean(average_distances)
    else:
        overall_average_distance = 0

    return overall_average_distance
# 定义信号处理函数
def signal_handler(sig, frame):
    print("\n录制结束！视频已保存为", output_filename)
    video_writer.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# 捕获 Ctrl+C 信号

def is_point_in_bbox(point, bbox):
    """判断毫米波点是否在 YOLO 检测框内"""
    x, y = point
    x1, y1, x2, y2 = bbox  # YOLO 框的左上角和右下角坐标
    return x1 <= x <= x2 and y1 <= y <= y2

def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算各自面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算 IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def enlarge_bbox(bbox, scale=1.2):
    """对目标框进行放大"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_x1 = int(x1 - (scale - 1) * width / 2)
    new_y1 = int(y1 - (scale - 1) * height / 2)
    new_x2 = int(x2 + (scale - 1) * width / 2)
    new_y2 = int(y2 + (scale - 1) * height / 2)
    return new_x1, new_y1, new_x2, new_y2

class DBSCANManual:
    def __init__(self, eps=5, min_samples=2):
        self.eps = eps  # 邻域范围
        self.min_samples = min_samples  # 最小样本数
        self.labels = []  # 聚类标签
        self.visited = []  # 是否被访问过
        self.noise = []  # 噪声点
        self.clusters = []  # 存储聚类结果

    def fit(self, points):
        n_points = len(points)
        self.labels = [-1] * n_points  # -1 表示未标记的点
        self.visited = [False] * n_points
        cluster_id = 0

        for i in range(n_points):
            if not self.visited[i]:
                self.visited[i] = True
                neighbors = self.region_query(points, i)
                if len(neighbors) < self.min_samples:
                    self.noise.append(i)  # 标记为噪声
                else:
                    self.expand_cluster(points, i, neighbors, cluster_id)
                    cluster_id += 1

        return self.labels

    def region_query(self, points, idx):
        """查找在给定点的eps邻域内的所有点"""
        neighbors = []
        for i, point in enumerate(points):
            if np.linalg.norm(np.array(point) - np.array(points[idx])) <= self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, points, idx, neighbors, cluster_id):
        """将邻域中的点扩展为一个聚类"""
        self.labels[idx] = cluster_id
        seeds = neighbors.copy()

        while seeds:
            current_point_idx = seeds.pop()
            if not self.visited[current_point_idx]:
                self.visited[current_point_idx] = True
                current_neighbors = self.region_query(points, current_point_idx)
                if len(current_neighbors) >= self.min_samples:
                    seeds.extend(current_neighbors)
            if self.labels[current_point_idx] == -1:
                self.labels[current_point_idx] = cluster_id

# 创建一个共享队列
data_queue = queue.Queue()
yolo_queue = queue.Queue()
radar_queue = queue.Queue()
global_a_count =0
global_non_motor_count = 0 
global_avg_speed = 0
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
        global global_a_count  # 声明使用全局变量
        global global_non_motor_count  # 声明使用全局变量
        global global_avg_speed  # 声明使用全局变量

        global_a_count = 0
        global_non_motor_count = 0
        global_avg_speed = 0
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
# 打印线程
def print_data_from_queue():
    last_yolo_time = 0
    last_radar_time = 0
    camera_frame = None
    camera_points = []

    while True:
        yolo_item = yolo_queue.get()
        if yolo_item is None:  # 如果遇到终止信号，退出循环
            print("no data")
            break
        # else:
            # print('get data')

        # yolo_timestamp = yolo_item["timestamp"]
        # yolo_data = yolo_item["data"]
        camera_frame = yolo_item['frame']


        original_height, original_width = camera_frame.shape[:2]
        # print(f"原始图像尺寸: 宽 {original_width}px, 高 {original_height}px")
        # 调整图像尺寸为 1440X810
        target_width, target_height = 1440, 810
        # target_width, target_height = 1920, 1080
        resized_frame = cv2.resize(camera_frame, (target_width, target_height))

        # 打印调整后的图像尺寸
        new_height, new_width = resized_frame.shape[:2]
        # print(f"调整后图像尺寸: 宽 {new_width}px, 高 {new_height}px")
        # tracks =yolov10_model.track(
        #         resized_frame, persist=False, show=False, verbose=False
        #     )
        cv2.imshow("Frame", resized_frame)

        video_writer.write(resized_frame)
        # 按下 'q' 键也可退出（作为备用方式）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        yolo_queue.task_done()
    video_writer.release()
    cv2.destroyAllWindows()

signal.signal(signal.SIGINT, signal_handler)


# 启动打印线程
print_thread = threading.Thread(target=print_data_from_queue)
print_thread.start()
url = "/home/buaa/sh/shangao-EventDetection/停车12.mp4"

# 示例：启动雷达处理线程
# radar_processor = RadarProcessor()
yolo_tracker = YOLOv10Tracker(yolov10_model)  
# radar_processor_thread = threading.Thread(target=radar_processor.start_receiving)
yolo_tracker_thread = threading.Thread(target=yolo_tracker.run_tracking, args=(url,))

# radar_processor_thread.start()
yolo_tracker_thread.start()
