import os
import shutil

import torch
import torch.nn as nn
import numpy as np
from util.utils import *
from dataset.facedata import get_pred, get_gallery
from PyQt5.QtWidgets import *
from MagFace.gen_feat import *
feat_list = './database/gallery_feat.list'


class Recognition(object):
    def __init__(self, gallery_path='.\\database', device=torch.device('cpu')):
        super(Recognition, self).__init__()

        self.gallery_path = gallery_path
        self.gallery_loader = None
        self.labelencoder = None
        self.gallery_features = None
        self.gallery_labels = None
        self.device = device
        self.gallery = None
        self.lines = None
        self._load_gallery()

    def _load_gallery(self):
        self.gallery_labels = []
        self.gallery_features = []
        # self.gallery_loader, self.labelencoder, self.gallery = get_gallery(gallery_path=self.gallery_path)
        # self.gallery_features, self.gallery_labels = extract_feature(self.gallery_loader, device=self.device)
        # self.gallery_labels = self.labelencoder.inverse_transform((self.gallery_labels.cpu()).numpy())

        with open(feat_list) as f:
            self.lines = f.readlines()
            for item in self.lines:
                ls = item.strip().split()
                feat = np.asarray(ls[-512:], dtype=np.float64)
                name = ls[:-512]
                if len(name) > 1:
                    label = '/'
                    for n in name:
                        label = label + ' ' + n
                else:
                    label = '/ ' + name[0]
                self.gallery_labels.append(label[2 - len(label):])
                self.gallery_features.append(feat)
        self.gallery_labels = np.vstack(self.gallery_labels).reshape(-1,)
        self.gallery_features = np.vstack(self.gallery_features)
        self.gallery_features = torch.from_numpy(self.gallery_features).to(self.device)

        # # 写文件
        # feats = self.gallery_features.cpu().numpy()
        # fio = open(feat_list, 'w')
        # for feat, person in zip(feats, self.gallery_labels):
        #     fio.write('{} '.format(person))
        #     for e in feat:
        #         fio.write('{} '.format(e))
        #     fio.write('\n')
        # fio.close()

    def _recog(self, img):
        pred_features = torch.from_numpy(Mag_gen_feat(img)).to(self.device)
        distance = rec_distance(pred_features, self.gallery_features)
        print(distance)
        pred_id = distance.argmin(dim=0)
        _nearest = distance.min(dim=0)
        return self.gallery_labels[pred_id], _nearest

    def _recog_1v1(self, img1, img2):
        pre_features1 = torch.from_numpy(Mag_gen_feat(img1)).to(self.device)
        pre_features2 = torch.from_numpy(Mag_gen_feat(img2)).to(self.device)
        dist = cos_distance(pre_features1, pre_features2)
        print(dist)
        if dist > 0.325:
            return False
        else:
            return True
