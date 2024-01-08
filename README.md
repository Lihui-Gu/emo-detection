# emo-detection
emotion detection ascend device &amp; aliyun esc

## env list

device:
```
numpy==1.22.4
onnxruntime==1.16.3
torch==1.13.0
torchvision==0.14.0
opencv-python==4.7.0.72
opencv-contrib-python==4.7.0.72
requests==2.28.1
```
cloud:
```
websockets==12.0
pysqlite3==0.5.2
pandas==2.1.4
Flask==2.2.5
```
## Quick Start

### train
```
bash scripts/train.sh
```

### run
```
# device
bash scripts/run_device.sh
# cloud
bash scripts/run_cloud.sh
# flask (cloud)
bash scrips/run_flask.sh
```

### Attention
Put `in.mp4` in `data/video/in.mp4`.
