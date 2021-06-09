import torch 
import cv2
import os
import time
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gaze_estimation.gaze_estimator.common import Face

def calculate_similarity(emb1, emb2, threshold=1.54):
    # assert(emb1.shape[0] == emb2.shape[0])
    assert(emb1.shape[1] == emb2.shape[1])

    diff = torch.sub(emb1, emb2)
    dist = torch.sum(torch.pow(diff, 2), dim=1)

    minimum, min_idx = torch.min(dist, dim=0) # dim=1
    # print(dist)
    
    # print(minimum, min_idx)
    # min_idx[minimum > threshold] = -1
    return min_idx, minimum


def draw_box_name(bbox, name, frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame

def parse_name(out_path, name='output'):
    name_idx = 0
    save_name = name + str(name_idx) + '.avi'
    while os.path.exists(os.path.join(out_path, save_name)):
        name_idx += 1
        save_name = name + str(name_idx) + '.avi'
    return os.path.join(out_path, save_name)


class Align_detector:

    def __init__(self, args):
        cfg = yaml.load(open(args.detector.dconfig), Loader=yaml.SafeLoader)
        gpu_mode = (args.device.detect == 'cuda')
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
        self.timer = Timer()
    
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

        ver_ave = np.mean(self.queue_ver, axis=0)

        x2 = np.max(ver_ave[0, :]).astype(np.int32)
        y2 = np.max(ver_ave[1, :]).astype(np.int32)
        x1 = np.min(ver_ave[0, :]).astype(np.int32)
        y1 = np.min(ver_ave[1, :]).astype(np.int32)    
        bbox = np.array([[x1, y1], [x2, y2]])
        landmarks = ver_ave#[:2, :].transpose(1, 0)

        leyes, reyes = ver_ave[:, 36:42], ver_ave[:, 42:48]
        leyes = np.mean(leyes, axis=1, keepdims=True)
        reyes = np.mean(reyes, axis=1, keepdims=True)

        misc = ver_ave[:, [30, 48, 54]]
        ver_ave = np.hstack((leyes, reyes, misc))

        # img_draw = cv_draw_landmark(self.queue_frame[self.n_pre], ver_ave)

        
        self.queue_ver.popleft()
        self.queue_frame.popleft()

        return [Face(bbox, landmarks[:2, :].transpose(1, 0))], ver_ave, landmarks
   

def get_point(target_size, font_size):

    return (target_size[1] // 2 - font_size[0] // 2, \
            target_size[0] // 2 - font_size[1] // 2) # W, H

# class Interface:
#     def __init__(self, cap):
#         self.target_shape = (cap[0] // 10, cap[1], 3) # (H, W)
#         self.target_shape = tuple(map(int, self.target_shape))
#         self.font = ImageFont.truetype('demo_util/gulim.ttc', self.target_shape[0] // 2)
#         self._generate_board()
#         self._color = (0, 0, 0, 0)
#         self.counter = 0
    
#     def failed_loop(self, msg='', t=0):
#         current = time.time()

#         while time.time() - current > t:
#             self.point = get_point(self.target_shape, self.font.get_size(txt))

#             board_pil = Image.fromarray(self.board)
#             draw = ImageDraw.Draw(board_pil)
#             draw.text(self.point, txt, font=self.font, fill=(0, 0, 0, 0))
            
#             frame_pil = Image.fromarray(frame)
#             txt = "Antispoofing Detected" if attack >= 0.6 else "Antispoofing is not Detected, 현재 : {}".format(name)
#             point = (self.target_shape[1] - self.font.getsize(txt)[0], self.target_shape[0])
#             draw = ImageDraw.Draw(frame_pil)
            
#             if id == name.split('_')[0]:
#                 draw.text(point, txt, font=self.font, fill=(0, 255, 0, 0))
#             else:
#                 draw.text(point, txt, font=self.font, fill=(0, 0, 255, 0))
#             frame = np.array(frame_pil)
            
#             frame[:self.target_shape[0], :self.target_shape[1]] = np.array(board_pil)

#             self._reset_board()

#     def _reset_board(self, ):
#         self.board[:] = (255, 255, 255)

class Text_to_Image:
    def __init__(self, cap, n_list):
        self.target_shape = (cap[0] // 10, cap[1], 3) # (H, W)
        self.target_shape = tuple(map(int, self.target_shape))
        self.font = ImageFont.truetype('demo_util/gulim.ttc', self.target_shape[0] // 2)
        self.font2 = ImageFont.truetype('demo_util/gulim.ttc', self.target_shape[0] // 2)
        self._generate_board()
        self._color = (0, 0, 0, 0)
        self.counter = 0
        self.n_list = n_list
        # self.point = (self.target_shape[1] // 2 - self.font.getsize(txt)[0] // 2, \
        #         self.target_shape[0] // 2 - self.font.getsize(txt)[1] // 2) # W, H
    
    def draw_mode(self, frame, name, id, txt = '시험 중, ', color=(0, 0, 0)):
        
        self.point = get_point(self.target_shape, self.font.getsize(txt))

        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)

        if id != name.split('_')[0]:
            color = (0, 0, 255)
            txt += '본인 인증 실패, 현재 {} 감지됨'.format(name.split('_')[0])
        else:
            txt += '본인 인증 성공'
         
        draw.text(self.point, txt, font=self.font, fill=color)
        
        # board = np.array(frame_pil)

        frame[:self.target_shape[0], :self.target_shape[1]] = np.array(frame_pil)

        return frame

    def easy_draw(self, frame, txt='', color=(0, 0, 0)): # frame, board -> frame + board (with text)

        self.point = get_point(self.target_shape, self.font.getsize(txt))

        board_pil = Image.fromarray(self.board)
        draw = ImageDraw.Draw(board_pil)
        draw.text(self.point, txt, font=self.font, fill=color)
        
        frame_pil = Image.fromarray(frame)
        
        frame[:self.target_shape[0], :self.target_shape[1]] = np.array(board_pil)

        self._reset_board()

        return frame 


    def draw(self, frame, remain, attack=0.0, name='', id=''): # frame, board -> frame + board (with text)

        txt1 = "{}초 동안 인증이 진행됩니다.".format(remain)
        self.point = get_point(self.target_shape, self.font.getsize(txt1))


        board_pil = Image.fromarray(self.board)
        draw = ImageDraw.Draw(board_pil)
        draw.text(self.point, txt1, font=self.font, fill=(0, 0, 0, 0))
        
        frame_pil = Image.fromarray(frame)

        color = None
        if attack >= 0.6:
            txt2 = "Antispoofing Detected / "
            color = (0, 0, 255)
        else:
            txt2 = "Antispoofing is not Detected / "
            color = (0, 255, 0)

        if id != name.split('_')[0]:
            txt2 = "{} 인식 실패, 현재 {} 인식됨".format(self.n_list[id], self.n_list[name.split('_')[0]])
            color = (0, 0, 255)
            self.counter -= 1
        else:
            txt2 += "{} 인식 성공".format(self.n_list[id])
            if color is None:
                color = (0, 255, 0) 
            self.counter += 1


        # txt = "Antispoofing Detected" if attack >= 0.6 else "Antispoofing is not Detected, 현재 : {}".format(name)
        draw = ImageDraw.Draw(frame_pil)
        
        # if id == name.split('_')[0]:
        draw.text(point, txt2, font=self.font, fill=color)
        # else:
        #     draw.text(point, txt2, font=self.font, fill=color)
        frame = np.array(frame_pil)
        
        frame[:self.target_shape[0], :self.target_shape[1]] = np.array(board_pil)

        self._reset_board()

        return frame, self.counter
    
    def await_for_time(self, thresh, msg='{}초 동안 대기합니다.', color=(0, 0, 0), out=None):
        cap = cv2.VideoCapture(-1)
        now = time.time()
        while time.time() - now < thresh:
            ok, frame = cap.read()
            self.point = get_point(self.target_shape, self.font.getsize(msg))

            broad_pil = Image.fromarray(self.board)
            draw = ImageDraw.Draw(broad_pil)
            draw.text(self.point, msg.format(int(thresh - (time.time() - now))), font=self.font, fill=color)

            # frame_pil = Image.fromarray(frame)
            # point = (self.target_shape[1] - self.font.getsize(msg.format(thresh - (time.time() - now)))[0], self.target_shape[0])
            # draw = ImageDraw.Draw(frame_pil)
            # draw.text(point, msg.format(thresh - (time.time() - now)), font=self.font, fill=(0, 255, 0, 0))
            # frame = np.array(frame_pil)

            frame[:self.target_shape[0], :self.target_shape[1]] = np.array(broad_pil)
            print(out)
            if out is not None:
                out.write(frame)
            cv2.imshow('Monitoring', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            self._reset_board()
        if out is not None:
            return out

        cap.release()

    def _generate_board(self,):
        self.board = np.zeros(self.target_shape, dtype=np.uint8)
        self.board[:] = (255, 255, 255)
    
    def _reset_board(self, ):
        self.board[:] = (255, 255, 255)

class Timer:
    def __init__(self, mode='list'):
        self.start = None
        self.mode = mode
        if mode == 'list':
            self.time_list = []
        elif mode == 'dic':
            self.time_dict = {}
    
    def tic(self,):
        assert self.start == None
        self.start = time.time()
    
    def toc(self,):
        assert self.start != None
        print(time.time() - self.start)
        if self.mode == 'list':
            self.time_list.append(time.time() - self.start)
        self.start = None
    
    def intergrate(self,):
        print('-' * 8)
        if self.mode == 'list':
            total = sum(self.time_list)
            for t in self.time_list:
                print('{}% {}, {}FPS'.format(100 * t / total, t, 1/t))
            print(total, 1/total)
            self.time_list = []
