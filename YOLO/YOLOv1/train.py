# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 11:27
# @Author  : aibc-hp
# @File    : train.py
# @Project : YOLOv1
# @Software: PyCharm

import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader

from backbone.vgg import vgg16, vgg16_bn
from backbone.resnet import resnet18, resnet50
from dataset.dataset import YoloDataset
from loss.yolo_loss import YoloLoss
# from visualization.visualize import Visualizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()  # True if there is a GPU and cuda is successfully installed

train_image_path = r'D:\object_detection\datasets\pascal_voc2007\voc_trainval\JPEGImages'  # 训练集图片的存储路径
train_image_name_path = r'D:\object_detection\datasets\pascal_voc2007\voc_trainval\trainval.txt'  # 训练集图片名称的存储路径（没有后缀名）
train_label_path = r'D:\object_detection\datasets\pascal_voc2007\voc_trainval\Annotations'  # 训练集图片中目标的类别及 ground-truth 信息的存储路径
test_image_path = r'D:\object_detection\datasets\pascal_voc2007\voc_test\JPEGImages'  # 测试集图片的存储路径
test_image_name_path = r'D:\object_detection\datasets\pascal_voc2007\voc_test\test.txt'  # 测试集图片名称的存储路径（没有后缀名）
test_label_path = r'D:\object_detection\datasets\pascal_voc2007\voc_test\Annotations'  # 测试集图片中目标的类别及 ground-truth 信息的存储路径

resnet18_pth = r'D:\object_detection\code\YOLO\YOLOv1\pretrained_pth\resnet18.pth'
resnet50_pth = r'D:\object_detection\code\YOLO\YOLOv1\pretrained_pth\resnet50.pth'
vgg16_pth = r'D:\object_detection\code\YOLO\YOLOv1\pretrained_pth\vgg16.pth'
vgg16_bn_pth = r'D:\object_detection\code\YOLO\YOLOv1\pretrained_pth\vgg16_bn.pth'

learning_rate = 0.001
num_epochs = 50
batch_size = 16
use_resnet = True
pretrained = [
              'resnet',
              # 'vgg',
              # 'best.pth'
             ]

if use_resnet:
    net = resnet50()
else:
    net = vgg16_bn()

print('===== Load pre-trained model =====')
if 'resnet' in pretrained:
    resnet = models.resnet50()
    resnet.load_state_dict(torch.load(resnet50_pth))
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
elif 'vgg' in pretrained:
    vgg = models.vgg16_bn()
    vgg.load_state_dict(torch.load(vgg16_bn_pth))
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and k.startswith('features'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
elif 'best.pth' in pretrained:
    net.load_state_dict(torch.load(r'./pth/best.pth'))

if torch.cuda.is_available():
    print('The current GPU is {}, the number of available GPUs is {}'.format(torch.cuda.current_device(), torch.cuda.device_count()))  # 当前使用的 GPU 和可用的 GPU 数量
else:
    print('The device does not have a GPU or cuda was not successfully installed')

criterion = YoloLoss(14, 2, 5, 0.5)  # 实例化 YoloLoss 对象；使用 vgg 且图像输入大小为 448*448 时，S=7；使用 resnet 且图像输入大小为 448*448 时，S=14

if use_gpu:
    net.cuda()

net.train()

# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr': learning_rate}]

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # 定义优化器
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

# 定义一个类别与数字的映射字典
category_dict = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

# 将图片转换成数值范围为 [0.0, 1.0] 且维度为 (C, H, W) 的张量
transform = transforms.Compose([transforms.ToTensor()])

# train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
train_dataset = YoloDataset(img_pth=train_image_path, imgname_pth=train_image_name_path, lab_pth=train_label_path, cate_dict=category_dict, img_size=448, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = YoloDataset(img_pth=test_image_path, imgname_pth=test_image_name_path, lab_pth=test_label_path, cate_dict=category_dict, img_size=448, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print('The train dataset has %d images' % (len(train_dataset)))
print('The test dataset has %d images' % (len(test_dataset)))

logfile = open(r'./log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='cornley')
best_test_loss = np.inf

print('===== Start training =====')
start_total = time.time()
for epoch in range(num_epochs):
    net.train()

    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('Epoch (%d / %d), LR is %f' % (epoch + 1, num_epochs, learning_rate))

    total_loss = 0.
    start_epoch = time.time()
    for i, (images, target) in enumerate(train_loader):
        start_iter = time.time()

        images = Variable(images)
        target = Variable(target)

        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)  # 前向传播
        loss = criterion(pred, target)  # 计算 loss
        total_loss += loss.item()  # 总的 loss
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 梯度反向传播
        optimizer.step()  # 参数更新

        iter_time = time.time() - start_iter  # 计算当前一次迭代所用的时间

        # 每 5 个 batch 输出一次信息
        if (i + 1) % 5 == 0:
            print('Train -- Epoch [%d/%d], Iter [%d/%d], current_iter_loss: %.4f, average_loss: %.4f, current_iter_time: %f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), total_loss / (i + 1), iter_time))
            num_iter += 1  # num_epochs * len(train_loader) 是整个训练过程中总共要迭代的次数
            # vis.plot_train_val(loss_train=total_loss / (i + 1))

    # validation
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        start_iter = time.time()

        images = Variable(images, volatile=True)
        target = Variable(target, volatile=True)

        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()  # 总的 loss

        iter_time = time.time() - start_iter  # 计算当前一次迭代所用的时间

        # 每 5 个 batch 输出一次信息
        if (i + 1) % 5 == 0:
            print('Test -- Epoch [%d/%d], Iter [%d/%d], current_iter_loss: %.4f, average_loss: %.4f, current_iter_time: %f'
                  % (epoch + 1, num_epochs, i + 1, len(test_loader), loss.item(), validation_loss / (i + 1), iter_time))

    validation_loss /= len(test_loader)  # 计算该 epoch 中平均一次迭代的 loss
    # vis.plot_train_val(loss_val=validation_loss)

    # if best_test_loss > validation_loss:
    #     best_test_loss = validation_loss
    #     print('Get best test loss %.5f' % best_test_loss)
    #     torch.save(net.state_dict(), f'./pth/best{epoch + 1}.pth')

    epoch_time = time.time() - start_epoch  # 计算训练当前 epoch 所用的时间
    print('Epoch (%d / %d) training completed, taking time: %f' % (epoch + 1, num_epochs, epoch_time))

    logfile.writelines('Epoch' + str(epoch + 1) + '\t' + 'validation_loss: ' + str(validation_loss) + '\n')
    logfile.flush()
    torch.save(net.state_dict(), f'./pth/epoch{epoch + 1}.pth')

logfile.close()

total_time = time.time() - start_total  # 计算整个训练所用的时间
print('Model training completed, taking time: %f' % total_time)

print('===== Training finished! =====')
