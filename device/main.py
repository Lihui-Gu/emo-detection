import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import onnxruntime
import torchvision.models as models
import argparse
from picsplit import PicSplit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emo_model", default="./model/emo_det.onnx")
    parser.add_argument("--split_model", default="./model/centerface.onnx")
    parser.add_argument("--video_path", default="./data/video/in.mp4")
    args = parser.parse_args()
    return args


def run(args):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = Image.open(image_path)
    device = torch.device('cpu')
    input_image = test_transform(input_image)
    input_image = input_image.unsqueeze(0)
    input_image = input_image.to(device).numpy()
    ort_session = onnxruntime.InferenceSession(args.emo_model)
    ort_inputs = {'input': input_image}
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





