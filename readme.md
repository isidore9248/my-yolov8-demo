# yolov8(GPU) + cuda11.8 + python3.9

## 环境搭建
### - 安装CUDA11.8支持的pytorch版本 (不要通过requirements.txt安装，否则为cpu版本)
``` bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```
- 如果在安装依赖时，提示需要t64.exe的问题，报错如下
```
ValueError: Unable to find resource t64.exe in package pip._vendor.distlib
```
- 解决方法：
    1. 删除损坏的 distlib 包
    ```
    rm -rf D:\your env path\your env name\Lib\site-packages\pip\_vendor\distlib
    ``` 
    2. 重新下载distlib 包 (使用阿里源)
    ```
    pip download distlib pip setuptools wheel -d . -i https://mirrors.aliyun.com/pypi/simple/    
    ```
    解压后手动复制```t64.exe```到
    ```
    D:\your env path\your env name\Lib\site-packages\pip\_vendor\distlib
    ```
- 网络问题安装失败
     </br>
    1. 离线安装包参考
    ```
    链接: https://pan.baidu.com/s/1KY-BjGf7pKIzWl46dXS4gw?pwd=dasb 提取码: dasb 
    ```
    2. 安装
    ```
    pip install xxx.whl
    ```

### 安装yolov8
```
pip install ultralytics
pip install ultralytics==8.2.21  # 指定版本避免自动升级到 v11:cite[5]:cite[8]
```
### 验证安装
- pytorchAndYolo.py 验证yolov8和pytorch安装
- pytorchTestGPU.py 验证GPU是否可用

### 导出pt文件为onnx文件
``` shell
yolo export model=best.pt format=onnx
```


## 文件结构

```
myTrainCode/
├── dataset/                     # 数据集文件夹
│   ├── train/                   # 训练集
│   │   ├── images/              # 训练图像
│   │   └── labels/              # 训练标签
│   ├── val/                     # 验证集
│   │   ├── images/              # 验证图像
│   │   └── labels/              # 验证标签
│   └── test/                    # 测试集
│       ├── images/              # 测试图像
│       └── labels/              # 测试标签
├── garbage_dataset.yaml         # 数据集配置文件
├── myyolo.py                    # YOLOv8 训练脚本
├── trainPlus.py                 # 改进后的训练脚本(添加异常处理机制)
├── mypredict.py                 # 预测脚本
├── split_dataset.py             # 数据集分割脚本
├── pytorchAndYolo.py            # 验证 PyTorch 和 YOLOv8 安装的脚本
├── pytorchTestGPU.py            # 验证 GPU 是否可用的脚本
├── readme.md                    # 项目说明文件
└── runs/                        # 训练结果保存目录
    ├── detect/                  # 检测任务结果
    └── train/                   # 训练任务结果
```


## 工具软件

- ### LabelImg
```
通过网盘分享的文件：windows_v1.8.1.zip
链接: https://pan.baidu.com/s/1Kch-U_LZezeYX5OQePqVag 提取码: dasb 
```