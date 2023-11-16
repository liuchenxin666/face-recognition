import math
import shutil

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from util.face_preprocess import preprocess
import os

mtcnn = MTCNN(device='cuda:0', keep_all=True)


# def extract_feature(img_loader, device=torch.device('cpu')):
#     # load backbone
#     checkpoint = './ArcFace/backbone_ir50_ms1m_epoch120.pth'
#     pretrained_dict = torch.load(checkpoint, map_location=device)
#     backbone = IR_50(input_size=[112, 112]).to(device)
#     backbone.load_state_dict(pretrained_dict)
#
#     features = None
#     labels = None
#     backbone.eval()
#     with torch.no_grad():
#         for x, y in img_loader:
#             x = x.to(device=device, dtype=torch.float32)
#             y = y.to(device=device)
#
#             feature = backbone(x)
#             if features is None:
#                 features = feature
#                 labels = y
#             else:
#                 features = torch.cat((features, feature), dim=0)
#                 labels = torch.cat((labels, y), dim=0)
#
#     return features, labels


def rec_distance(pred, gallery):
    assert pred.size() == (1, 512) and gallery.size(1) == 512
    d = torch.zeros([gallery.size(0)], dtype=torch.float64)
    for idx in range(gallery.size(0)):
        d[idx] = cos_distance(pred, gallery[idx].unsqueeze(dim=0))
    # d = (gallery - pred) ** 2
    # d = torch.sqrt(torch.sum(d, dim=1))
    return d


def cos_distance(features0, features1):
    assert features0.size(0) == 1 and features1.size(0) == 1
    
    # Distance based on cosine similarity
    dot = torch.sum(torch.mul(features0, features1))  # 按行
    norm = torch.sqrt(torch.sum(features0 * features0)) * torch.sqrt(torch.sum(features1 * features1))  # 默认是二范数
    # shaving
    similarity = torch.clamp(dot / norm, -1., 1.)
    dist = torch.acos(similarity) / torch.tensor(math.pi)  # 余弦距离 cosθ的θ,单位是pai
    return dist


def my_detect(img):
    # Returns
    # 0 : 没有人脸
    # 1 : 一张人脸
    # 2 : 多于一张
    bboxes, probs, points = mtcnn.detect(img=img, landmarks=True)
    
    if bboxes is None:
        return 0
    
    if bboxes.shape[0] > 1:
        return 2
    else:
        return bboxes, points


def my_align(img, bbox, points):
    # 相似性变换，把倾斜的人脸对齐
    image = preprocess(img, bbox[0], points[0].squeeze(), image_size="112,112")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


if __name__ == '__main__':
    pass
