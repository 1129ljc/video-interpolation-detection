'''
@Time    : 2021/9/3 9:42
@Author  : ljc
@FileName: dataload.py
@Software: PyCharm
'''

import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


transform = transforms.Compose(
    [
        # transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


class DataSet(Dataset):
    def __init__(self, data_file_path):
        super(Dataset, self).__init__()
        self.json_file_path = data_file_path
        assert os.path.isfile(data_file_path), print('The dataset json file cannot be read')
        with open(data_file_path, 'r', encoding='utf8')as fp:
            data = fp.readlines()
        self.image_path_list = []
        self.image_label_list = []
        for i in range(len(data)):
            line = data[i].split(' ')
            self.image_path_list.append(line[0])
            self.image_label_list.append(int(line[1][0:-1]))
        self.image_num = len(self.image_path_list)

    def __len__(self):
        return self.image_num

    def __getitem__(self, item):
        image_file = self.image_path_list[item]
        label = self.image_label_list[item]
        label_torch = torch.tensor(label)
        # image_torch = transform(Image.open(image_file).convert('RGB'))
        image_torch = torch.from_numpy(np.array(Image.open(image_file).convert('RGB')))
        image_torch = image_torch.permute(2, 0, 1).float()
        image_torch = torch.unsqueeze(image_torch[0, :, :], dim=0)
        return image_file, image_torch, label_torch
