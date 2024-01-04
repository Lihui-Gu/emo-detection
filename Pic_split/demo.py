import os
import cv2
from Pic_split import Pic_split


def main():
    video_path = 'in.mp4'
    sp = Pic_split()
    sp.run(video_path)
    while True:
        frame = sp.get()#ndarray，HxWxd格式，值的范围是0-255，d按RGB顺序
        if frame is None:
            break
        # do sth here


if __name__ == '__main__':
    main()
