
import torch
import torch.backends.cudnn as cudnn
from RetinaFace.data import *
from RetinaFace.models.retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def retinaface_loader(conf, device):

    cfg = None   
    if conf['network'] == "mobile0.25":
        cfg = cfg_mnet
        trained_model = './RetinaFace/weights/mobilenet0.25_Final.pth'
    elif conf['network'] == "resnet50":
        cfg = cfg_re50
        trained_model = './RetinaFace/weights/Resnet50_Final.pth'
    elif conf['network'] == "mobilenetv2":
        cfg = cfg_mnet2
        trained_model = './RetinaFace/weights/mobilenetv2_Final.pth'
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, trained_model, conf['cpu'])
    net.eval()
    print('Finished loading RetinaFace + {} model!'.format(conf['network']))
    cudnn.benchmark = True
    net = net.to(device)

    return net, cfg
