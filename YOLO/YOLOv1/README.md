YOLOv1

- dataset -- 该目录存放了处理数据集的代码文件
  - dataset.py -- 处理数据集

- backbone -- 该目录存放了主干网络的代码文件
  - resnet.py -- 包含 resnet18、resnet34、resnet50、resnet101、resnet152 的网络结构
    - 使用了 resnet18 与 resnet50 来训练和测试模型
    - 与原始的 resnet 模型相比，删除了 stage 5 后面的全局池化层与全连接层，增加了一个带有空洞卷积的 block，类似 DetNet 网络，旨在增加网络感受野而不改变输出的特征图大小
    - 输入的图像大小为 $448*448$ 时，最终输出的特征图大小为 $14*14$ 
  - vgg.py -- 包含 vgg11、vgg11_bn、vgg13、vgg13_bn、vgg16、vgg16_bn、vgg19、vgg19_bn 的网络结构
    - 使用了 vgg16 与 vgg16_bn 来训练和测试模型
    - 输入的图像大小为 $448*448$ 时，最终输出的特征图大小为 $7*7$ 
- loss -- 该目录存放了处理 loss 的代码文件
  - yolo_loss.py -- 计算 loss；通过 mask 选取对应部分，分别计算 mse loss，最后按照公式加起来
- visualization -- 该目录存放了可视化训练 loss 的代码文件
  - visualize.py -- 可视化训练 loss
    - 通过 visdom 进行可视化；由于内网限制，在模型的训练和测试中，并没有使用到可视化

- image_enhancement_effect_visualization -- 该目录存放了经过数据增强的效果图片
- test_images -- 该目录存放了用于测试模型的图片，也用于存储测试得到的结果图片
- pretrained_pth -- 该目录用于存放 resnet18、resnet50 等模型的预训练权重文件，这些预训练权重文件可通过 resnet.py 和 vgg.py 中的 url 获得
- pth -- 该目录用于存储训练过程中每个 epoch 得到的模型参数文件 
- eval_results -- 该目录存放 voc2007 数据集评估模型的结果

- train.py -- 训练模型
- predict.py -- 测试模型，可使用单张图片进行推理测试
- eval_voc2007.py -- 评估模型，使用 voc2007 数据集来评估训练好的模型

 

训练注意事项：

- 不需要手动将数据放到 CPU 或 GPU 上，如果有 GPU 并成功安装 cuda，代码会自动选择将数据移动到 GPU 上进行 cuda 加速计算

- 如果使用 resnet50 或其他模型，请确保 pretrained_pth 目录中有 resnet50.pth 或其他模型的预训练权重文件
- 默认使用 resnet50 模型，如果使用 resnet18，则需将 resnet.py 中 `self._make_detnet_layer(in_planes=2048)` 的参数值改为 512
- 如果要使用 vgg 模型，需要把 train.py 中的 `use_resnet = True` 改为 False，并把 pretrained 列表中的 'resnet' 注释掉，改用 'vgg'
- 如果要使用 visdom 进行可视化，需要把 `import visdom`、`vis.plot_train_val()` 等语句恢复



训练（train.py）：

- 确保有原始的 trainval.txt，代码运行后该文件会发生改动
- 将 train.py 中的 train_image_path、train_image_name_path、train_label_path、test_image_path、test_image_name_path、test_label_path、resnet50_pth 修改为自己的路径



预测（predict.py）：

- 修改以下路径为自己的即可
  - model.load_state_dict(torch.load(r'./pth/epoch48.pth'))
  - image_name = './test_images/dog.jpg'
  - cv2.imwrite(r'./test_images/dog_result.jpg', image)
- predict_gpu 函数中，选择 image = cv2.imread(root_path + image_name)



评估（eval_voc2007.py）：

- 确保有原始的 test.txt，代码运行后该文件会发生改动
- 修改以下路径为自己的即可
  - model.load_state_dict(torch.load(r'./pth/epoch48.pth'))
- predict_gpu 函数中，选择 image = cv2.imread(root_path + '/' + image_name)

