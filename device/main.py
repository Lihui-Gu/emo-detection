import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import onnxruntime
import torchvision.models as models
import argparse
import threading
from picsplit import PicSplit
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emo_model", default="./model/emo_det.onnx")
    parser.add_argument("--split_model", default="./model/centerface.onnx")
    parser.add_argument("--video_path", default="./data/video/in.mp4")
    args = parser.parse_args()
    return args


def run(args):
    sp = PicSplit(args.split_model)
    def run_pic_split():
        sp.run(args.video_path)
    thread = threading.Thread(target=run_pic_split)
    thread.start()
    print("===============")
    while True:
        frame = sp.get()
        if frame is None:
            print("frame is none, sleep 1s ...")
            time.sleep(1)
            continue
        ort_session = onnxruntime.InferenceSession(args.emo_model)
        ort_inputs = {'input': frame}
        ort_output = ort_session.run(None,ort_inputs)[0]
        print(ort_output.argmax(axis=1)[0])

'''
0 -> angry
1 -> disgust
2 -> fear
3 -> happy
4 -> neutral
5 -> sad
6 -> surprise
'''
if __name__ == "__main__":
    args = parse_args()
    run(args)





