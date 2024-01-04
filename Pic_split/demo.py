import os
import cv2
from Pic_split import Pic_split


def main():
    video_path = 'in.mp4'
    sp = Pic_split()
    sp.run(video_path)
    while True:
        frame = sp.get()
        if frame is None:
            break
        # do sth here


if __name__ == '__main__':
    main()
