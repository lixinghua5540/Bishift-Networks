# coding=utf-8
import torch
# import numpy as np
from options.train_options import TrainOptions
import numpy as np
import cv2
import util.util as util
import os
from PIL import Image
import glob
import itertools
import random
#####用于生成图像数据矩形mask   或者多边形mask

mask_folder = 'masks/testing_masks'     #生成的mask保存的地址
test_folder = './datasets/tmp/test'  #从该目录下的图像中生成mask
util.mkdir(mask_folder)

opt = TrainOptions().parse()   #获取配置参数 options文件中的配置

f = glob.glob(test_folder+'/*.png') # 获取所有的png图像文件
#这里设置框的大小
began_x = 50
began_y = 50
w = 80
h = 80

for fl in f:

    print('Generating mask for test image: ' + os.path.basename(fl))
    # 保存mask到指定的目录
    # b  = np.array([[[50,50],  [50,450], [450,450],[450,50]]], dtype = np.int32) #不仅仅可以为矩形，可以为任何多边形 多个点的坐标连接构成的形状

    b = np.array([[[began_x, began_y], [began_x + w, began_y], [began_x + w, began_y + h], [began_x, began_y + h]]],
                 dtype=np.int32)
    im = np.zeros((256, 256), dtype="uint8")  # 这里设置mask图像大小
    cv2.fillPoly(im, b, 255)  # 填充

    # random_list = list(itertools.product(range(1, 256), range(1, 256)))
    # defect_ratio = 0.1   #缺失率
    # defect_num = int(256 * 256 * defect_ratio)     #缺失数
    # b = random.sample(random_list, defect_num)
    #
    # im = np.zeros((256, 256), dtype="uint8")  # 这里设置mask图像大小
    # for i in range(defect_num):
    #     im[b[i]] = 255

    cv2.imwrite(os.path.join(mask_folder, os.path.splitext(os.path.basename(fl))[0] + '_mask.png'),im)





