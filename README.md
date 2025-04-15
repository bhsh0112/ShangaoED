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

