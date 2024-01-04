from torchvision import transforms
from PIL import Image
import onnxruntime
import argparse
import threading
from picsplit import PicSplit
import time
import socket
import json

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
       
        server_ip = '127.0.0.1'
        server_port = 12345

        data = {"message": ort_output.argmax(axis=1)[0]}
        json_data = json.dumps(data)

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        client_socket.connect((server_ip, server_port))

        client_socket.sendall(json_data.encode('utf-8'))

    client_socket.close()

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





