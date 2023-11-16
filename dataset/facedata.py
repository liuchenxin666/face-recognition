import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import cv2
# from skimage import io
feat_list = './gallery_feat.list'


class facedata(Dataset):
    def __init__(self, data, labels):
        super(facedata, self).__init__()

        self.data = data
        self.labels = labels
        self.transforms = T.Compose([
            # T.ToPILImage(),
            # T.Resize(size=(112, 112)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ])

    def __getitem__(self, index):
        return self.transforms(self.data[index]), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)


def get_gallery(gallery_path):
    person_list = os.listdir(gallery_path)

    gallery_data = []   # gallery图像
    gallery_labels = []     # 对应人名
    for person in person_list:
        imgs_path = os.path.join(gallery_path, person)
        imgs = os.listdir(imgs_path)
        for img in imgs:
            img_path = os.path.join(imgs_path, img)
            data = cv2.imread(img_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            gallery_data.append(data)
            gallery_labels.append(person)

    # labelencoder 对标签进行编码， fit_transform把标签变为Int,  inverse_transform变换回标签
    labelencoder = LabelEncoder()
    gallery_labels = labelencoder.fit_transform(gallery_labels)

    gallery = facedata(gallery_data, gallery_labels)
    # gallery_labels已经变成tensor了，不再是int
    gallery_loader = DataLoader(gallery, batch_size=32, shuffle=False, drop_last=False)
    return gallery_loader, labelencoder, gallery


def get_pred(imgs):
    if not isinstance(type(imgs), list):     # isinstance 判断是不是同样类型
        imgs = [imgs]
    N = len(imgs)
    fake_labels = np.zeros(N)               # 虚假的标签
    pred = facedata(imgs, fake_labels)
    pred_loader = DataLoader(pred, batch_size=32, shuffle=False, drop_last=False)
    return pred_loader
