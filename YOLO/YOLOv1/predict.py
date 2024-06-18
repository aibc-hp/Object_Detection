# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 9:46
# @Author  : aibc-hp
# @File    : predict.py
# @Project : YOLOv1
# @Software: PyCharm

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from backbone.vgg import vgg16, vgg16_bn
from backbone.resnet import resnet50


VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def decoder(pred):
    """
    :param pred: A tensor -- (1, 14, 14, 30)
    """
    grid_num = 14  # 最终输出的特征图一行或一列的单元格数量
    boxes = []
    cls_indexes = []
    probs = []
    cell_size = 1. / grid_num  # 单元格大小
    pred = pred.data  # 获取 pred 张量中的数据
    pred = pred.squeeze(0)  # (14, 14, 30)

    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain > 0.1  # 大于阈值
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it > 0.9
    mask = (mask1 + mask2).gt(0)
    # min_score, min_index = torch.min(contain, 2)  # 每个 cell 只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell 左上角
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image

                    # [cx, cy, w, h] -> [x1, y1, x2, y2]
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexes.append(cls_index)
                        probs.append(contain_prob * max_prob)

    cls_indexes = [x.unsqueeze(0) if x.numel() == 1 else x for x in cls_indexes]
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexes = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)  # (n, 4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indexes = torch.cat(cls_indexes, 0)  # (n,)

    keep = nms(boxes, probs)

    return boxes[keep], cls_indexes[keep], probs[keep]


def nms(bboxes, scores, threshold=0.25):
    """
    :param bboxes: tensor -- [N, 4]
    :param scores: tensor -- [N,]
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:  # order.numel() 计算 order 张量中所有元素的总数
        order = order.view(-1)
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.LongTensor(keep)


def predict_gpu(model, image_name, root_path=''):
    # image = cv2.imread(root_path + '/' + image_name)  # 以 BGR 形式读取一张图片；执行 eval_voc2007.py 时使用该语句
    image = cv2.imread(root_path + image_name)  # 以 BGR 形式读取一张图片；执行 predict.py 时使用该语句
    h, w = 0, 0
    if image is None:
        print(f"Error loading image {root_path + '/' + image_name}")
    else:
        h, w, _ = image.shape  # opencv 读取的图片维度为 (H, W, C)
    img = cv2.resize(image, (448, 448))  # 修改尺寸 -> (448, 448)；默认使用双线性插值法
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    mean = (123, 117, 104)  # 数据集在 RGB 各通道上的均值
    img = img - np.array(mean, dtype=np.float32)  # RGB 各通道减去均值
    transform = transforms.Compose([transforms.ToTensor()])  # 将数组转换成张量，并进行归一化（所有值除以 255）和维度变换（(H, W, C) -> (C, H, W)）
    img = transform(img)
    img = Variable(img[None, :, :, :], volatile=True)  # Variable 用于表示一个有状态的张量（比如梯度），但是从 0.2.0 版本开始，Variable 类被弃用，取而代之的是直接使用 torch.Tensor 类，并且增加了对自动求导的支持；img 张量通过 None 增加一个维度，img[None, :, :, :] 与 img.unsqueeze(0) 等效；volatile=True 表示不进行梯度跟踪，与 torch.no_grad() 等效，用于模型推理
    if torch.cuda.is_available():
        img = img.cuda()  # 将 img 张量移动到 GPU 上

    pred = model(img)  # torch.Size([1, 14, 14, 30])
    if torch.cuda.is_available():
        pred = pred.cpu()  # 将 pred 张量移动到 CPU 上
    boxes, cls_indexs, probs = decoder(pred)

    result = []
    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])

    return result


if __name__ == '__main__':
    model = resnet50()  # 实例化模型对象

    print('load model...')
    model.load_state_dict(torch.load(r'./pth/epoch48.pth'))  # 模型加载各网络层权重
    model.eval()  # 将模型设置为评估模式；在评估模式下，模型中的特定层（如 Dropout 和 Batch Normalization）会以不同的方式运行，以便于进行准确的评估和测试；在评估模式下，Dropout 层不会随机丢弃神经元，而是使用所有神经元；在评估模式下，Batch Normalization 层将使用整个批次的统计数据来计算均值和方差，而不是使用单个批次的统计数据

    if torch.cuda.is_available():
        model.cuda()  # 将模型的参数和缓冲区移动到 GPU 上，以便利用 CUDA 进行加速计算；在调用 model.cuda() 之前，确保相关数据也已经被移动到 GPU 上
    else:
        model.cpu()  # 将模型的参数和缓冲区移动到 CPU 上；在调用 model.cpu() 之前，确保相关数据也已经被移动到 CPU 上

    image_name = './test_images/dog.jpg'
    # image_name = './test_images/person.jpg'
    image = cv2.imread(image_name)  # 以 BGR 形式读取测试图片

    print('predicting...')
    result = predict_gpu(model, image_name)
    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv2.imwrite(r'./test_images/dog_result.jpg', image)
    # cv2.imwrite(r'./test_images/person_result.jpg', image)
