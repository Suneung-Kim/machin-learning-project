#!/usr/bin/env python

from typing import Optional

import datetime
import logging
import pathlib
import argparse
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('./3DDFA')

import imageio 
import cv2
import numpy as np
import yacs.config
from perceptron_gaze.per_model import Model

from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)
from gaze_estimation.utils import load_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator
from detect import Align_detector

from temp import Text_to_Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)
        self.detector = Align_detector(config)

        # self.cap = self._create_capture()
        self.cap = self._get_reader()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def run(self) -> None:
        # while True:
        model = Model()
        model.to(self.config.device)
        model.load_state_dict(torch.load("./perceptron_gaze/checkpoint/best_not_roll/ckpt.pth"))
        text1="집중도"
        terminate_t = 0
        total_count = 0
        normal_count = 0
        abnormal_count = 0
        concentrated_rate = 0
        color=(0,0,255)
        text = ""

        drawer = Text_to_Image((480,640))
        
        for i, frame_bgr in enumerate(self.cap):
            frame = frame_bgr[..., ::-1]
            if i == 0:
                self.detector.first_frame(frame)
            else:
                self.detector.middle_frame(frame)

                if self.config.demo.display_on_screen:
                    self._wait_key()
                    if self.stop:
                        break

                # ok, frame = self.cap.read()
                # if not ok:
                #     break

                undistorted = cv2.undistort(
                    frame, self.gaze_estimator.camera.camera_matrix,
                    self.gaze_estimator.camera.dist_coefficients)

                self.visualizer.set_image(frame.copy())
                faces = self.detector.detect(frame.copy())
                # print("faces :   {}".format(len(faces[0].landmarks)))
                # print("face.model3d : {}".format(faces[0].model3d))
    
                # faces = self.gaze_estimator.detect_faces(undistorted)
                
                for face in faces:
                    self.gaze_estimator.estimate_gaze(undistorted, face)
                    self._draw_face_bbox(face)
                    head_yaw, head_pitch, distance = self._draw_head_pose(face)
                    self._draw_landmarks(face)
                    self._draw_face_template_model(face)

                    h_yaw = head_yaw
                    h_pitch = head_pitch
                    h_distance = distance
                    
                    # f = open('./perceptron_gaze/gaze_data/train/train.txt','a')
                    e_yaw, e_pitch = self._draw_gaze_vector(face)
                    
                    # e_yaw, e_pitch = self._draw_gaze_vector(face)
                    r_yaw = e_yaw[0]
                    r_pitch = e_pitch[0]
                    l_yaw = e_yaw[1]
                    l_pitch = e_pitch[1]
                    value = [round(h_yaw,2), round(h_pitch,2), round(h_distance,2), round(r_yaw,2), round(r_pitch,2), 
                              round(l_yaw,2), round(l_pitch,2)]
                    
                    self._display_normalized_image(face)


                if len(value)==0:
                  print("값이 없습니다.")
                else:
                    total_count+=1
                    # print(self.config.device)
                    inputs = torch.tensor(value)
                    inputs = inputs.to(self.config.device, dtype=torch.float)
                    # print(inputs)
                    output = model(inputs)
                    # print("output : ", output)
                    _, predicted = output.max(0)
                    print(output)
                    predicted_action = predicted
                    # print("predicted: ", predicted_action)
                    if predicted_action == 0:
                        if abnormal_count >= 2:
                            text = "경고"
                            color = (0,0,255)

                        abnormal_count += 1
                    else:                            
                        normal_count+=1
                        abnormal_count = 0
                        text = ""
                        color = (0,0,255)            
                concentrated_rate = round(float((normal_count/total_count)*100),2)



                if self.config.demo.use_camera:
                    self.visualizer.image = self.visualizer.image[:, ::-1]
                if self.writer:
                    self.writer.write(self.visualizer.image)
                if self.config.demo.display_on_screen:
                    frame = self.visualizer.image
                    frame = np.array(frame)
                    frame = drawer.easy_draw(frame, text, color=(0,0,255))

                    start_t = time.time()
                    sec = start_t - terminate_t
                    terminate_t = start_t
                    FPS = int(1./sec)
                    # cv2.putText(frame, "FPS : {}".format(str(FPS)), (40,40),cv2.FONT_HERSHEY_PLAIN,3,color,3)
                    # cv2.putText(frame, "{}".format(text), (40,50),cv2.FONT_HERSHEY_PLAIN,4,color,3)
                    # cv2.putText(frame, "focus_rate : {}".format(concentrated_rate), (320,40),cv2.FONT_HERSHEY_PLAIN,2,color,2)
                    # frame = cv2.line(frame,(320, 240),(320, 240),color,5)
                    cv2.imshow('frame', frame)

        self.cap.release()
        if self.writer:
            self.writer.release()

    def _create_capture(self) -> cv2.VideoCapture:
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap
    
    def _get_reader(self):
        return imageio.get_reader("<video0>")

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        output_path = self.output_dir / f'{self._create_timestamp()}.{ext}'
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> None:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')
        return yaw, pitch, face.distance

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        yaw_app=[]
        pitch_app=[]

        length = self.config.demo.gaze_visualization_length
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
                yaw_app.append(yaw)
                pitch_app.append(pitch)

        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

        return yaw_app, pitch_app


def main():
    # parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')

    # args = parser.parse_args()

    config = load_config()
    demo = Demo(config)
    demo.run()


if __name__ == '__main__':
    main()
