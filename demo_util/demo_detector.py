from PIL import Image
from demo_util.align_trans import get_reference_facial_points, warp_and_crop_face
from demo_util.detector import detect_faces
import numpy as np
import cv2 
import torch

import sys
sys.path.append('./RetinaFace')

from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.box_utils import decode, decode_landm
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms


def mtcnn_align(conf, frame):
    
    img = Image.fromarray(frame)
    reference = get_reference_facial_points(default_square = True) * conf['INPUT_SIZE'][0] / 112.
    try:
        bbox, landmarks = detect_faces(img)
    except Exception as e:
        print(e)
        return None, None, None
    if len(landmarks) == 0:
        print('no landmark')
        return None, None, None

    # coords = bbox[0].astype(np.int32)

    # frame = frame[coords[0]:coords[2], coords[1]:coords[3]]
    # frame = cv2.resize(frame, (112, 112))
    # frame = Image.fromarray(frame)

    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, (conf['INPUT_SIZE'][0], conf['INPUT_SIZE'][1]))
    # img_warped = Image.fromarray(warped_face)
    return bbox, warped_face, facial5points

def align_module(frame, net, cfg, cfg2, device):
    
    reference = get_reference_facial_points(default_square = True) * 1.

    faces = []

    img = frame.copy()
    img = np.float32(img)

    im_height, im_width, _ = img.shape
            
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg2['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg2['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > cfg2['confidence_threshold'])[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds] 

    # keep top-K before NMS
    # order = scores.argsort()[::-1][:1]
    order = scores.argsort()[::-1]#[:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, cfg2['nms_threshold'])
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)

    for i, land in enumerate(landms):
        # print(scores[i])
        facial5points = [[land[j * 2], land[j * 2 + 1]] for j in range(5)]
        warped_face = warp_and_crop_face(frame, facial5points, reference, (112, 112))
        faces.append(warped_face)
        

    # facial5points = [[landms[0][j * 2], landms[0][j * 2 + 1]] for j in range(5)]
    # warped_face = warp_and_crop_face(frame, facial5points, reference, (112, 112))
    # img_warped = Image.fromarray(warped_face)
    # cv2.imshow('', warped_face)
    # cv2.waitKey(0)
    return dets, faces, np.concatenate((dets, landms), axis=1)