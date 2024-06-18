# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 10:00
# @Author  : aibc-hp
# @File    : dataset.py
# @Project : YOLOv1
# @Software: PyCharm

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class YoloDataset(Dataset):
    def __init__(self, img_pth: str, imgname_pth: str, lab_pth: str, cate_dict: dict, img_size: int, train: bool, transform: transforms.Compose):
        self.img_pth = img_pth
        self.imgname_pth = imgname_pth
        self.lab_pth = lab_pth
        self.cate_dict = cate_dict
        self.img_size = img_size
        self.train = train
        self.transform = transform
        self.img_name = []  # 用于存储训练集图片的名称；共计 5011 个元素（样本）
        self.gt = []  # 用于存储训练集图片中目标的 ground-truth 信息；共计 5011 个元素（样本），15662 个目标
        self.label = []  # 用于存储训练集图片中目标的类别信息；共计 5011 个元素（样本），15662 个目标
        self.mean = (123, 117, 104)  # 所有图片各个通道（RGB）的均值

        # 获取所有图片中目标的 ground-truth 及类别信息
        xml_list = os.listdir(self.lab_pth)
        xml_data = []  # 初始化一个列表，用于存储所有 xml 文件中的 ground-truth 及类别信息；三维列表
        for filename in xml_list:
            xml_file_path = os.path.join(self.lab_pth, filename)  # xml 文件路径
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            objects_data = []  # 初始化一个列表，用于存储一个 xml 文件中的 ground-truth 及类别信息；二维列表
            for obj in root.findall('object'):
                name = obj.find('name').text  # 提取类别名称
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)  # 提取左上角点 x 坐标
                ymin = int(bndbox.find('ymin').text)  # 提取左上角点 y 坐标
                xmax = int(bndbox.find('xmax').text)  # 提取右下角点 x 坐标
                ymax = int(bndbox.find('ymax').text)  # 提取右下角点 y 坐标

                objects_data.append([xmin, ymin, xmax, ymax, self.cate_dict[name]])

            xml_data.append(objects_data)

        # 读取 trainval.txt 文件中的所有图片名称
        with open(self.imgname_pth, 'r') as f:
            lines = f.readlines()

        # 对所有的图片名称添加 .jpg 后缀，并在名称后面添加 ground-truth 及类别信息，然后保存到原文件
        with open(self.imgname_pth, 'w') as f:
            for line in lines:
                new_line = line.rstrip() + '.jpg' + ' '
                for single_target in xml_data[lines.index(line)]:
                    new_line += ' '.join(map(str, single_target))
                    new_line += ' '
                new_line += '\n'
                f.write(new_line)

        # 读取修改后的 trainval.txt 文件，并进行相应处理
        with open(self.imgname_pth, 'r') as f:
            lines = f.readlines()
            for line in lines:
                split_data = line.strip().split()
                self.img_name.append(split_data[0])  # 将图片名称添加到 self.img_name 中

                num_gt = (len(split_data) - 1) // 5
                gt = []
                label = []
                for i in range(num_gt):
                    xmin = float(split_data[1 + 5 * i])
                    ymin = float(split_data[2 + 5 * i])
                    xmax = float(split_data[3 + 5 * i])
                    ymax = float(split_data[4 + 5 * i])
                    c = split_data[5 + 5 * i]

                    gt.append([xmin, ymin, xmax, ymax])
                    label.append(int(c))

                self.gt.append(torch.Tensor(gt))  # 将 ground-truth 信息转换成张量，再添加到 self.gt 中
                self.label.append(torch.LongTensor(label))  # 将类别信息转换成张量，再添加到 self.label 中

        self.num_samples = len(self.gt)  # 共计 5011 个元素（样本）

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        img = cv2.imread(os.path.join(self.img_pth, img_name))
        gt_boxes = self.gt[idx].clone()
        labels = self.label[idx].clone()

        if self.train:
            # 可视化图片处理效果
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.subplot(121)
            # for box in gt_boxes:
            #     cv2.rectangle(img, (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])), color=(255, 0, 0))
            # plt.axis('off')
            # plt.imshow(img)
            # img, gt_boxes = self.random_flip(img, gt_boxes)
            # plt.subplot(122)
            # for box in gt_boxes:
            #     cv2.rectangle(img, (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])), color=(255, 0, 0))
            # plt.axis('off')
            # plt.imshow(img)
            # plt.show()

            # img = self.random_bright(img)  # 随机改变图片的明暗度
            img, gt_boxes = self.random_flip(img, gt_boxes)  # 随机左右翻转图片
            img, gt_boxes = self.random_scale(img, gt_boxes)  # 随机缩放图片；固定图片高度，在 (0.8, 1.2) 范围内随机生成一个浮点数，作为宽度的缩放系数
            img = self.random_blur(img)  # 随机对图片进行均值滤波；滤波核大小为 (5, 5)
            img = self.random_brightness(img)  # 随机改变图片的亮度；亮度系数从 [0.5, 1.5] 中随机选择一个
            img = self.random_hue(img)  # 随机改变图片的色相；色相系数从 [0.5, 1.5] 中随机选择一个
            img = self.random_saturation(img)  # 随机改变图片的饱和度；饱和度系数从 [0.5, 1.5] 中随机选择一个
            img, gt_boxes, labels = self.random_shift(img, gt_boxes, labels)  # 随机对图片进行平移变换
            img, gt_boxes, labels = self.random_crop(img, gt_boxes, labels)  # 随机裁剪图片

        h, w, _ = img.shape
        gt_boxes /= torch.Tensor([w, h, w, h]).expand_as(gt_boxes)  # 将绝对坐标值转换为相对于图像宽高的坐标值
        img = self.BGR2RGB(img)
        img = self.sub_mean(img, self.mean)  # 图片各通道减去均值
        img = cv2.resize(img, (self.img_size, self.img_size))  # 将图片 resize 成指定大小
        target = self.encoder(gt_boxes, labels)  # 获取大小为 (num_grid, num_grid, 30) 的张量
        img = self.transform(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_scale(self, im, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = im.shape
            im = cv2.resize(im, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return im, boxes
        return im, boxes

    def random_blur(self, im):
        if random.random() < 0.5:
            im = cv2.blur(im, (5, 5))
        return im

    def random_brightness(self, im):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(im)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            im = self.HSV2BGR(hsv)
        return im

    def random_hue(self, im):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(im)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            im = self.HSV2BGR(hsv)
        return im

    def random_saturation(self, im):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(im)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            im = self.HSV2BGR(hsv)
        return im

    def random_shift(self, im, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = im.shape
            after_shift_image = np.zeros((height, width, c), dtype=im.dtype)
            after_shift_image[:, :, :] = (104, 117, 123)  # bgr 各通道均值
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shift_image[int(shift_y):, int(shift_x):, :] = im[:height - int(shift_y), :width - int(shift_x), :]
            elif shift_x >= 0 and shift_y < 0:
                after_shift_image[:height + int(shift_y), int(shift_x):, :] = im[-int(shift_y):, :width - int(shift_x), :]
            elif shift_x < 0 and shift_y >= 0:
                after_shift_image[int(shift_y):, :width + int(shift_x), :] = im[:height - int(shift_y), -int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shift_image[:height + int(shift_y), :width + int(shift_x), :] = im[-int(shift_y):, -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)

            if len(boxes_in) == 0:
                return im, boxes, labels

            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shift_image, boxes_in, labels_in
        return im, boxes, labels

    def random_crop(self, im, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = im.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, w, h = int(x), int(y), int(w), int(h)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)

            if len(boxes_in) == 0:
                return im, boxes, labels

            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)
            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)
            labels_in = labels[mask.view(-1)]
            img_croped = im[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return im, boxes, labels

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def sub_mean(self, im, mean):
        mean = np.array(mean, dtype=np.float32)
        im = im - mean
        return im

    def encoder(self, boxes, labels):
        grid_num = 14  # 最终输出的特征图一行或一列的单元格数量
        target = torch.zeros((grid_num, grid_num, 30))  # 初始化一个大小为 (num_grid, num_grid, 30) 的张量，用于存储每个单元格的目标信息；30 表示每个单元格对应的 2 个预测框信息和置信度以及 20 个类别的概率
        cell_size = 1. / grid_num  # 单元格大小
        wh = boxes[:, 2:] - boxes[:, :2]  # 得到 boxes 的宽和高
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # 得到 boxes 的中心点坐标

        # 遍历每个 box 的中心点坐标
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]  # 取出第 i 个 box 的中心点坐标
            ij = (cxcy_sample / cell_size).ceil() - 1  # 得到目标中心点对应的单元格索引 ij；ij[0] 表示列索引（从 0 开始编号），对应 cx；ij[1] 表示行索引（从 0 开始编号），对应 cy
            target[int(ij[1]), int(ij[0]), 4] = 1  # 中心点落在该单元格，表示该单元格包含目标，故该单元格对应的第一个预测框置信度置为 1
            target[int(ij[1]), int(ij[0]), 9] = 1  # 中心点落在该单元格，表示该单元格包含目标，故该单元格对应的第二个预测框置信度置为 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1  # 中心点落在该单元格，表示该单元格包含目标，故该单元格对应的类别置信度置为 1
            xy = ij * cell_size  # 匹配到的单元格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size  # 中心点相对于匹配单元格左上角的偏移量
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]  # 第一个预测框的宽和高
            target[int(ij[1]), int(ij[0]), :2] = delta_xy  # 第一个预测框的中心点偏移量
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]  # 第二个预测框的宽和高
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy  # 第二个预测框的中心点偏移量

        return target


def main():
    image_path = r'D:\object_detection\datasets\pascal_voc2007\voc_trainval\JPEGImages'  # 训练集图片的存储路径
    image_name_path = r'D:\object_detection\datasets\pascal_voc2007\voc_trainval\trainval.txt'  # 训练集图片名称的存储路径（没有后缀名）
    label_path = r'D:\object_detection\datasets\pascal_voc2007\voc_trainval\Annotations'  # 训练集图片中目标的类别及 ground-truth 信息的存储路径

    # 定义一个类别与数字的映射字典
    category_dict = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

    # 将图片转换成数值范围为 [0.0, 1.0] 且维度为 (C, H, W) 的张量
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = YoloDataset(img_pth=image_path, imgname_pth=image_name_path, lab_pth=label_path, cate_dict=category_dict, img_size=448, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    train_iter = iter(train_loader)
    for i in range(100):
        img, target = next(train_iter)
        print(img, target)


if __name__ == '__main__':
    main()


