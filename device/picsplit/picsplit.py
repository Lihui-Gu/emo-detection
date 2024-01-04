import cv2
import scipy.io as sio
import os
from centerface import CenterFace
from torchvision import transforms

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PicSplit:
    def __init__(self):
        self.is_save = False
        self.ls = []
        self.score_bar = 0.75
        self.step = 600

    def test_image(self, frame, path):
        h, w = frame.shape[:2]
        landmarks = True
        centerface = CenterFace(landmarks=landmarks)
        if landmarks:
            dets, lms = centerface(frame, h, w, threshold=0.35)
        else:
            dets = centerface(frame, threshold=0.35)
        cnt = 0
        for det in dets:
            boxes, score = det[:4], det[4]
            if score < self.score_bar:
                continue
            p = 0
            u = max(0, int(boxes[1]) - p)
            d = min(h, int(boxes[3]) + p)
            l = max(0, int(boxes[0]) - p)
            r = min(w, int(boxes[2]) + p)
            # print(u / h, d / h, l / w, r / w)
            cropped_image = frame[u:d, l:r]
            # (path + str(cnt) + '.jpg')
            if self.is_save:
                cv2.imwrite(path + str(cnt) + '.jpg', cropped_image)
            self.ls.append(cropped_image)
            cnt += 1
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        if landmarks:
            for lm in lms:
                for i in range(0, 5):
                    cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
        # cv2.imshow('out', frame)
        cv2.waitKey(0)
        return cnt

    def run(self, video_path):
        self.ls = []
        video = cv2.VideoCapture()
        if not video.open(video_path):
            frame = cv2.imread(video_path)
            cnt = self.test_image(frame, 'pic/')
            print('emo_count:{}'.format(cnt))
        else:
            x = 0
            idx = 0
            while True:
                _, frame = video.read()
                if frame is None:
                    break
                if x % self.step == 0:
                    if self.is_save:
                        save_path = os.path.join('pic/' + str(idx) + '.jpg')
                        cv2.imwrite(save_path, frame)
                        if not os.path.exists('pic/' + str(idx)):
                            os.makedirs('pic/' + str(idx))
                    cnt = self.test_image(frame, 'pic/' + str(idx) + '/')
                    idx += 1
                    print('frame{}:emo_count:{}'.format(idx, cnt))
                x += 1
            video.release()
        self.ls.reverse()

    def get(self):
        if self.ls:
            img = self.ls.pop()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        return None


