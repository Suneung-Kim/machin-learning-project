import torch
import numpy as np 
import os
import cv2
from torchvision import transforms as trans

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def load_faces(conf):
    embs = torch.load(os.path.join(conf['FACE_ROOT'], 'facebank.pth'))
    names = np.load(os.path.join(conf['FACE_ROOT'], 'names.npy'))
    return embs, names

def prepare_faces(conf, backbone, tta=True, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    backbone.eval()
    embeddings = []
    names = []
    for p in os.listdir(conf['FACE_ROOT']):
        if not os.path.isdir(os.path.join(conf['FACE_ROOT'], p)):
            continue
        else:
            embs = []
            for f in os.listdir(os.path.join(conf['FACE_ROOT'], p)):
                print(os.path.join(conf['FACE_ROOT'], p, f))
                try:
                    img = cv2.imread(os.path.join(conf['FACE_ROOT'], p, f))
                except:
                    continue
                if img.size != (112, 112):
                    img = cv2.resize(img, (112, 112))
                
                img = img[..., ::-1]

                # flip image horizontally
                flipped = cv2.flip(img, 1)

                # load numpy to tensor
                img = img.swapaxes(1, 2).swapaxes(0, 1)
                img = np.reshape(img, [1, 3, 112, 112])
                img = np.array(img, np.float32)
                img = (img - 127.5) / 128.0
                img = torch.from_numpy(img)

                flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
                flipped = np.reshape(flipped, [1, 3, 112, 112])
                flipped = np.array(flipped, np.float32)
                flipped = (flipped - 127.5) / 128.0
                flipped = torch.from_numpy(flipped)
                
                with torch.no_grad():
                    if tta:
                        emb_batch = backbone(img.to(device)).cpu() + backbone(flipped.to(device)).cpu()
                        features = l2_norm(emb_batch)
                    else:
                        features = l2_norm(backbone(img.to(device)).cpu())
                embs.append(features)
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(p)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(conf['FACE_ROOT'], 'facebank.pth'))
    np.save(os.path.join(conf['FACE_ROOT'], 'names.npy'), names)
    return embeddings, names
