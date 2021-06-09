import torch
import cv2
import numpy as np
import os

import matplotlib as plt

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def extract_feature(img, backbone, device, tta=True):
    
    resized = cv2.resize(img, (112, 112))
    resized = resized[..., ::-1]

    flipped = cv2.flip(resized, 1)

    resized = resized.swapaxes(1, 2).swapaxes(0, 1)
    resized = np.reshape(resized, [1, 3, 112, 112])
    resized = np.array(resized, dtype=np.float32)
    resized = (resized - 127.5) / 128.0
    resized = torch.from_numpy(resized)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype=np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)


    with torch.no_grad():
        if tta:
            emb_batch = backbone(resized.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(resized.to(device)).cpu())
    
    return features