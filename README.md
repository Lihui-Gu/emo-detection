# emo-detection
emotion detection ascend device &amp; aliyun esc

## env list

```
numpy==1.22.4
onnxruntime==1.16.3
torch==1.13.0
torchvision==0.14.0
opencv-python==4.7.0.72
opencv-contrib-python==4.7.0.72
requests==2.28.1
```

## run

### train
```
bash scripts/train.sh
```
### inference
```
# device
bash scripts/run_device.sh
# cloud
bash scripts/run_cloud.sh
```

### folder structure
├─dataset(表情识别数据集)             
│  ├─test
│  │  ├─angry
│  │  ├─disgust
│  │  ├─fear
│  │  ├─happy
│  │  ├─neutral
│  │  ├─sad
│  │  └─surprise
│  └─train
│      ├─angry
│      ├─disgust
│      ├─fear
│      ├─happy
│      ├─neutral
│      ├─sad
│      └─surprise
├─main(训练、测试代码入口)
│  └─result
├─sample(需要识别的图片放这里)
├─scripts
└─test
    └─send-receive

