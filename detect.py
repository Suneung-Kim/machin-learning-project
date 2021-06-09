import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque

import sys
sys.path.append('./3DDFA')

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark

from gaze_estimation.gaze_estimator.common import Face

class Align_detector:

    def __init__(self, args):
        cfg = yaml.load(open(args.detector.dconfig), Loader=yaml.SafeLoader)
        gpu_mode = (args.device == 'cuda')
        self.face_boxes = FaceBoxes()
        self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        self.queue_ver = deque()
        self.queue_frame = deque()
        self.n_pre, self.n_next = args.detector.n_pre, args.detector.n_next
        self.n = self.n_pre + self.n_next + 1
        self.param_lst = None
        self.roi_box_lst = None
        self.dense_flag = args.detector.opt in ('2d_dense', '3d')
        self.pre_ver = None
    
    def first_frame(self, frame):
        # the first time, detect face, here we only use the first face, you can change on your need
        boxes = self.face_boxes(frame)
        boxes = [boxes[0]]
        self.param_lst, self.roi_box_lst = self.tddfa(frame, boxes)
        ver = self.tddfa.recon_vers(self.param_lst, self.roi_box_lst, dense_flag=self.dense_flag)[0]

        # refine
        self.param_lst, self.roi_box_lst = self.tddfa(frame, [ver], crop_policy='landmark')
        ver = self.tddfa.recon_vers(self.param_lst, self.roi_box_lst, dense_flag=self.dense_flag)[0]

        # padding queue
        for _ in range(self.n_pre):
            self.queue_ver.append(ver.copy())
        self.queue_ver.append(ver.copy())

        for _ in range(self.n_pre):
            self.queue_frame.append(frame.copy())
        self.queue_frame.append(frame.copy())
        
        self.pre_ver = ver

    def middle_frame(self, frame):
        self.param_lst, self.roi_box_lst = self.tddfa(frame, [self.pre_ver], crop_policy='landmark')
        roi_box = self.roi_box_lst[0]

        if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
            boxes = self.face_boxes(frame)
            boxes = [boxes[0]]
            self.param_lst, self.roi_box_lst = self.tddfa(frame, boxes)
        
        ver = self.tddfa.recon_vers(self.param_lst, self.roi_box_lst, dense_flag = self.dense_flag)[0]
        
        self.queue_ver.append(ver.copy())
        self.queue_frame.append(frame.copy())

        self.pre_ver = ver

    def detect(self, frame):
        if len(self.queue_ver) < self.n:
            return frame
        
        detected = []

        ver_ave = np.mean(self.queue_ver, axis=0)

        x2 = np.max(ver_ave[0, :]).astype(np.int32)
        y2 = np.max(ver_ave[1, :]).astype(np.int32)
        x1 = np.min(ver_ave[0, :]).astype(np.int32)
        y1 = np.min(ver_ave[1, :]).astype(np.int32)  

        bbox = np.array([[x1, y1], [x2, y2]])
        landmarks = ver_ave[:2, :].transpose(1, 0)

        # leyes, reyes = ver_ave[:, 36:42], ver_ave[:, 42:48]
        # leyes = np.mean(leyes, axis=1, keepdims=True)
        # reyes = np.mean(reyes, axis=1, keepdims=True)

        # misc = ver_ave[:, [30, 48, 54]]
        # ver_ave = np.hstack((leyes, reyes, misc))

        # img_draw = cv_draw_landmark(self.queue_frame[self.n_pre], ver_ave)
        
        self.queue_ver.popleft()
        self.queue_frame.popleft()
        detected.append(Face(bbox, landmarks))

        return detected

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='3DDFA/configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()

    detect = Align_detector(args)

    reader = imageio.get_reader("<video0>")
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]
        if i == 0:
            detect.first_frame(frame_bgr)
        else:
            detect.middle_frame(frame_bgr)
        
        frame = detect.run(frame_bgr)

        cv2.imshow('image', frame)
        if cv2.waitKey(20) & 0xff == ord('q'):
            break
