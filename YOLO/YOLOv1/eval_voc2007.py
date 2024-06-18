# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 9:18
# @Author  : aibc-hp
# @File    : eval_voc2007.py
# @Project : YOLOv1
# @Software: PyCharm

import os

import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from predict import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def get_label(lab_pth):
    # 定义一个类别与数字的映射字典
    category_dict = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
                     'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                     'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

    # 获取所有图片中目标的 ground-truth 及类别信息
    xml_list = os.listdir(lab_pth)
    xml_data = []  # 初始化一个列表，用于存储所有 xml 文件中的 ground-truth 及类别信息；三维列表
    for filename in xml_list:
        xml_file_path = os.path.join(lab_pth, filename)  # xml 文件路径
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

            objects_data.append([xmin, ymin, xmax, ymax, category_dict[name]])

        xml_data.append(objects_data)

    return xml_data


def transform_txt(imgname_pth, xml_data):
    # 读取 test.txt 文件中的所有图片名称
    with open(imgname_pth, 'r') as f:
        lines = f.readlines()

    # 对所有的图片名称添加 .jpg 后缀，并在名称后面添加 ground-truth 及类别信息，然后保存到原文件
    with open(imgname_pth, 'w') as f:
        for line in lines:
            new_line = line.rstrip() + '.jpg' + ' '
            for single_target in xml_data[lines.index(line)]:
                new_line += ' '.join(map(str, single_target))
                new_line += ' '
            new_line += '\n'
            f.write(new_line)


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap calculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mprec = np.concatenate(([0.], prec, [0.]))

        for i in range(mprec.size - 1, 0, -1):
            mprec[i - 1] = np.maximum(mprec[i - 1], mprec[i])

        j = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[j + 1] - mrec[j]) * mprec[j + 1])

    return ap


def voc_eval(preds, target, voc_classes=VOC_CLASSES, threshold=0.5, use_07_metric=False):
    """
    :param preds: {'cat': [[image_name, confidence, x1, y1, x2, y2], ...], 'dog': [[image_name, confidence, x1, y1, x2, y2], ...], ...}
    :param target: {(image_name, class): [[x1, y1, x2, y2], ], ...}
    """
    aps = []
    for i, class_ in enumerate(voc_classes):
        pred = preds[class_]  # [[image_name, confidence, x1, y1, x2, y2], ...]
        if len(pred) == 0:  # 如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('The ap of class {} is {}'.format(class_, ap))
            aps += [ap]
            break

        image_names = [x[0] for x in pred]  # 出现该类别的所有图片名称
        confidences = np.array([float(x[1]) for x in pred])  # 预测为该类别的所有边界框的置信度
        bboxes = np.array([x[2:] for x in pred])  # 预测为该类别的所有边界框

        # sorted by confidence
        sorted_ind = np.argsort(-confidences)  # np.argsort() 返回数组值从小到大排序后的索引值；使用 -confidences 进行降序排序
        bboxes = bboxes[sorted_ind, :]  # bboxes 根据得到的降序索引重新排列
        image_names = [image_names[x] for x in sorted_ind]  # image_names 根据得到的降序索引重新排列

        # go down dets and mark TPs and FPs
        npos = 0.  # 正样本数量
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])  # 统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_names)  # 预测出的边界框数量
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for idx, image_name in enumerate(image_names):
            bbox = bboxes[idx]  # 预测框
            if (image_name, class_) in target:
                gts = target[(image_name, class_)]  # [[x1, y1, x2, y2], ]
                for gt in gts:
                    # compute overlaps
                    ixmin = np.maximum(gt[0], bbox[0])
                    iymin = np.maximum(gt[1], bbox[1])
                    ixmax = np.minimum(gt[2], bbox[2])
                    iymax = np.minimum(gt[3], bbox[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) + (gt[2] - gt[0] + 1.) * (gt[3] - gt[1] + 1.) - inters
                    if union == 0:
                        print(bbox, gt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[idx] = 1
                        gts.remove(gt)  # 这个框已经匹配到了，不能再匹配
                        if len(gts) == 0:
                            del target[(image_name, class_)]  # 删除已经没有 gt 的键值对

                        break

                fp[idx] = 1 - tp[idx]
            else:
                fp[idx] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print('The ap of class {} is {}'.format(class_, ap))
        aps += [ap]
    print('mAP: {}'.format(np.mean(aps)))


def test_eval():
    preds = {
        'cat': [['image01', 0.9, 20, 20, 40, 40], ['image01', 0.8, 20, 20, 50, 50], ['image02', 0.8, 30, 30, 50, 50]],
        'dog': [['image01', 0.78, 60, 60, 90, 90]]
    }
    target = {
        ('image01', 'cat'): [[20, 20, 41, 41]],
        ('image01', 'dog'): [[60, 60, 91, 91]],
        ('image02', 'cat'): [[30, 30, 51, 51]]
    }
    voc_eval(preds, target, voc_classes=['cat', 'dog'])


if __name__ == '__main__':
    # test_eval()  # 测试程序是否无误

    image_path = r'D:\object_detection\datasets\pascal_voc2007\voc_test\JPEGImages'  # 测试集图片的存储路径
    image_name_path = r'D:\object_detection\datasets\pascal_voc2007\voc_test\test.txt'  # 测试集图片名称的存储路径（没有后缀名）
    label_path = r'D:\object_detection\datasets\pascal_voc2007\voc_test\Annotations'  # 测试集图片中目标的类别及 ground-truth 信息的存储路径

    target = defaultdict(list)  # 创建一个带有默认值的字典，访问字典中不存在的键时，会使用默认值，而不会抛出错误；这里使用空列表 [] 作为所有新键的默认值
    preds = defaultdict(list)
    image_list = []  # 存储图片名称（带后缀名）

    xml_data = get_label(label_path)  # 获取所有图片中目标的 ground-truth 及类别信息
    transform_txt(image_name_path, xml_data)  # 对所有的图片名称添加 .jpg 后缀，并在名称后面添加 ground-truth 及类别信息，然后保存到原文件

    # 获取测试集图片名称（没有后缀名）
    f = open(image_name_path)
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited_data = line.strip().split()
        file_list.append(splited_data)
    f.close()

    print('===== prepare ground-truth target =====')
    for index, image_file in enumerate(file_list):
        image_name = image_file[0]
        image_list.append(image_name)

        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])

            class_name = VOC_CLASSES[c-1]
            target[(image_name, class_name)].append([x1, y1, x2, y2])

    # print(image_list)  # ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg', '000006.jpg', '000008.jpg', ...]
    # print(target)  # {('000001.jpg', 'dog'): [[48, 240, 195, 371]], ('000001.jpg', 'person'): [[8, 12, 352, 498]], ('000002.jpg', 'train'): [[139, 200, 207, 301]], ...}

    print('===== start predict =====')
    # model = vgg16()
    # model.classifier = nn.Sequential(
    #         nn.Linear(512 * 7 * 7, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 1470),
    #     )
    model = resnet50()
    model.load_state_dict(torch.load(r'./pth/epoch48.pth'))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    for img_name in tqdm(image_list):
        result = predict_gpu(model, img_name, root_path=image_path)  # result[[left_up, right_bottom, cls_name, img_name, prob], ...]
        for (x1, y1), (x2, y2), class_name, image_name, prob in result:
            preds[class_name].append([image_name, prob, x1, y1, x2, y2])

        # image = cv2.imread(image_path + img_name)
        # for left_up, right_bottom, class_name, _, prob in result:
        #     color = Color[VOC_CLASSES.index(class_name)]
        #     cv2.rectangle(image, left_up, right_bottom, color, 2)
        #     label = class_name + str(round(prob, 2))
        #     text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        #     p1 = (left_up[0], left_up[1]- text_size[1])
        #     cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        #     cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
        #     cv2.imshow('img', image)
        #     cv2.waitKey(0)

    print('===== start evaluate =====')
    voc_eval(preds, target, voc_classes=VOC_CLASSES)
