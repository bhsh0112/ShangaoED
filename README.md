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

### 多事件同时发生时

- 堵车与停车同时发生

  在当前逻辑下，不堵车是“停车”时间发生的前提，因此不存在二者同时发生的情况

- 堵车与行人同时发生/停车与行人同时发生

​		二者同时发生时会同时输出二者的提示信息

​	综上，下表展示了所有时间情况及其输出信息

| 事件     | 无事件           | 堵车  | 停车                   | 行人              | 堵车+行人                    | 停车+行人                                     |
| -------- | ---------------- | ----- | ---------------------- | ----------------- | ---------------------------- | --------------------------------------------- |
| 输出信息 | everything is ok | jam！ | there are parked cars! | there are people! | jam！<br />there are people! | there are parked cars!<br />there are people! |

## 代码修改

### 1 修改输出方式

​		**当前，输出方式为在图片左上角输出文字，如需修改请参考本节**

​		修改位置：`EventDet.oy`中的EventDetector的output方法（91~105行）

```python
def output(self,judger):
		if judger.result[0]:
        if judger.result[2]:
        		#同时发生“行人事件”和“堵车事件”的输出
        else:
          	#只发生“堵车事件”的输出
    else:
        if judger.result[1]:
            if judger.result[2]:
            		#同时发生“停车事件”和“行人事件”的输出
            else:
              	#只发生“停车事件”的输出
        else:
            if judger.result[3]:#people
                #只发生“行人事件”的输出
            else:
                #无事件发生的输出
```

## TODO

