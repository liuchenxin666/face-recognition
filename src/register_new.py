import os
import cv2
from util.utils import *
from src.Face_quality import FIQA
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from MagFace.gen_feat import *
from PIL import Image
import numpy as np

gallery_path = './database'
feat_list = './database/gallery_feat.list'


def Register(form, filename, id, R):
    # 读取中文路径
    register_img = Image.open(filename)
    register_img = cv2.cvtColor(np.asarray(register_img), cv2.COLOR_RGB2BGR)
    # register_img = cv2.imread(filename)
    detected = my_detect(register_img)
    if detected == 0:
        QMessageBox.information(form, '提醒', '未检测到人脸！')
        return 0
    elif detected == 2:
        QMessageBox.information(form, '提醒', '有多张人脸！')
        return 0
    else:
        bbox, points = detected
        aligned_img = my_align(register_img, bbox, points)
        score, threshold = FIQA(aligned_img)
        print(score)
        if score <= 27.56:
            QMessageBox.information(form, '提醒', '人脸图像质量较低，注册失败！')
            return 0
        else:
            img_path = gallery_path + '/' + id
            print(img_path)
            if id not in os.listdir(gallery_path):
                os.mkdir(img_path)
            seq = len(os.listdir(img_path)) + 1
            # image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            feature_register = Mag_gen_feat(aligned_img)
            
            # cv2.imwrite(img_path + '/' + str(id) + '_' + str(seq) + '.bmp', aligned_img)
            aligned_img = Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
            aligned_img.save(img_path + '/' + str(id) + '_' + str(seq) + '.bmp')
            
            with open(feat_list, 'a+') as fio:
                # fio.write('{} '.format(id))
                fio.write(id + ' ')
                for e in feature_register[0]:
                    fio.write('{} '.format(e))
                fio.write('\n')
            R._load_gallery()
            return 1


if __name__ == "__main__":
    pass
