## Get Start

### 1 创建虚拟环境

```
conda create -n ED python=3.10
conda activate ED
```

### 2 软件环境安装

#### 2.1 安装基础软件环境

```
pip install -r requirments.txt
```

#### 2.2 安装pytorch

在官网（https://pytorch.org/ ）查找适配设备的版本安装

#### 2.3 安装yolov10软件包

```
pip install -q git+https://github.com/THU-MIG/yolov10.git
```

#### 2.4 运行代码

```
python EventDet.py --source /path/to/vedio(file->video 0->webcam) --output /path/to/output weights /path/to/weights_file
```

## 事件检测逻辑

### 1 停车检测

同时满足以下条件：

- 未发生”堵车“
- 存在任一车辆连续两帧速度小于阈值（该阈值目前设置为检测框宽度的0.3倍）

### 2 堵车检测

同时满足以下条件：

- 同时存在n辆速度小于阈值的车辆

## TODO

