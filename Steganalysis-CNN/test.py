'''
@Time    : 2021/9/3 13:51
@Author  : ljc
@FileName: test.py
@Software: PyCharm
'''

import os
import cv2
import tqdm
import torch
import datetime
import torchvision
import torch.nn as nn
from resnet import _ResNet18
from dataload import DataSet
import torch.optim as optim
from torch.utils.data import DataLoader


def test():
    # 各种参数
    json_file_test = '/sdb1/ljc/interpolation_detection/dataset/AVI_FPS30/AOBMC/FPS_30_AOBMC_CIF_test.txt'
    trained_model = '/sdb1/ljc/interpolation_detection/dataset/AVI_FPS30/AOBMC/train_log1/40.pth'


    # 加载网络
    print('加载网络')
    print('torch.cuda.is_available()', torch.cuda.is_available())
    print('torch.cuda.device_count()', torch.cuda.device_count())
    print('torch.cuda.get_device_name()', torch.cuda.get_device_name())
    model = torch.load(f=trained_model)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    print(model)

    # 加载数据
    print('加载测试数据')
    test_dataset = DataSet(json_file_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=4)
    test_dataset_size = len(test_dataset)
    test_dataset_batch_size = len(test_loader)
    print('test_dataset_size: ', test_dataset_size)
    print('test_dataset_batch_size: ', test_dataset_batch_size)

    num = 0
    for idx, (image_torch, label_torch) in enumerate(test_loader):
        image_torch = image_torch.to(device)
        predict = model(x=image_torch)
        _, predict_label = torch.max(predict, dim=1)
        label_torch = label_torch.cpu().numpy()
        predict_label = predict_label.cpu().numpy()
        print(label_torch, predict_label)
        if label_torch == predict_label:
            num = num + 1
    print(num / test_dataset_size)


if __name__ == '__main__':
    test()
