'''
@Time    : 2021/9/2 15:17
@Author  : ljc
@FileName: train.py
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


def train():
    # 各种参数
    lr_init = 1e-4
    epochs = 40
    batch_size = 4
    train_file_txt = '/sdb1/ljc/interpolation_detection/dataset/AVI_FPS30/AOBMC/FPS_30_AOBMC_CIF_train.txt'
    train_log_file_dir = '/sdb1/ljc/interpolation_detection/dataset/AVI_FPS30/AOBMC/train_log3/'

    # 加载网络
    print('Load the network')
    print('torch.cuda.is_available()', torch.cuda.is_available())
    print('torch.cuda.device_count()', torch.cuda.device_count())
    print('torch.cuda.get_device_name()', torch.cuda.get_device_name())
    # resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18 = _ResNet18()
    device = torch.device('cuda:0')
    resnet18.to(device)
    print('resnet18', resnet18)

    # 加载数据
    print('Loading data training')
    train_dataset = DataSet(train_file_txt)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    train_dataset_size = len(train_dataset)
    train_dataset_batch_size = len(train_loader)
    print('train_dataset_size: ', train_dataset_size)
    print('train_dataset_batch_size: ', train_dataset_batch_size)

    # 设置损失函数优化器等
    print('Set up the loss function optimizer, etc')
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=resnet18.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1, last_epoch=-1)

    # 训练
    for epoch_index in range(epochs):
        resnet18.train()
        epoch_train_loss = 0.0
        epoch_train_loss1 = 0.0
        for idx, (image_torch, label_torch) in enumerate(train_loader):
            image_torch, label_torch = image_torch.to(device), label_torch.to(device)
            optimizer.zero_grad()
            predict = resnet18(x=image_torch)
            loss = loss_function(input=predict, target=label_torch.long())
            loss.backward()
            optimizer.step()
            iter_loss = loss.cpu().float()
            epoch_train_loss = epoch_train_loss + iter_loss
            epoch_train_loss1 = epoch_train_loss1 + iter_loss
            time = datetime.datetime.now().strftime('%H:%M:%S')
            if idx % 100 == 0:
                epoch_train_loss /= 100
                log_string = '[%s] epoch:[%d] iter:[%d]/[%d] loss:[%.5f]' % (
                    time, epoch_index + 1, idx + 1, train_dataset_batch_size, iter_loss)
                print(log_string)
                epoch_train_loss = 0.0
        scheduler.step()
        log_string = 'epoch:[%d] train_loss:[%.5f]' % (epoch_index + 1, epoch_train_loss1)
        print(log_string)
        torch.save(resnet18, train_log_file_dir + str(epoch_index + 1) + '.pth')


if __name__ == '__main__':
    train()
