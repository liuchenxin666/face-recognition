from MagFace.models.network_inf import builder_inf, NetworkBuilder_inf
import cv2
from torchvision import transforms
import torch
import argparse
import numpy as np
import json
from tqdm import tqdm

# parse the args
parser = argparse.ArgumentParser(description='Trainer for posenet')
parser.add_argument('--arch', default='iresnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--resume',
                    default='./MagFace/models/Backbone_Epoch_71_checkpoint.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
args = parser.parse_args()


def load_state_dict(model, state_dict):
    # model_dict = model.state_dict()
    # pretrained_dict = {'module.' + k: v for k, v in state_dict.items() if ('module.' + k) in model_dict.keys() and v.size() == model_dict['module.' + k].size()}
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    
# # 加载magface预训练的模型
# model = builder_inf(args)

# 加载我重新训练的模型
model = NetworkBuilder_inf(args)
ck = torch.load(args.resume, map_location='cuda:0')
load_state_dict(model, ck)


model = model.to('cuda:0')
trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
    ])


def Mag_get_score(img):
    _feat = Mag_gen_feat(img)
    return np.linalg.norm(_feat)


def Mag_gen_feat(img):
    model.eval()
    data = torch.randn(1, 3, 112, 112)
    data[0, :, :, :] = trans(img)
    data = data.to('cuda:0')
    embedding_feat = model(data)
    _feat = embedding_feat.data.cpu().numpy()
    return _feat


if __name__ == '__main__':
    json_path = 'D:/Database/CASIA-aug/train-test.json'
    with open(json_path, 'r') as f:
        ret = json.load(f)
    imgs_path = ret['test']['imgs_path']
    scores = []
    for i in tqdm(range(len(imgs_path))):
        img = cv2.imread(imgs_path[i])
        scores.append(Mag_get_score(img))
    print('magnitude is in [' + str(min(scores)) + ', ' + str(max(scores)) + ']')
    pass
