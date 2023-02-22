import torch
# import numpy as np
from options.train_options import TrainOptions
import util.util as util
import os
from PIL import Image
import glob

#####用于生成图像数据的mask

mask_folder = 'masks/testing_masks'     #生成的mask保存的地址
test_folder = './datasets/exp/test/200320'  #从该目录下的图像中生成mask
util.mkdir(mask_folder)

opt = TrainOptions().parse()   #获取配置参数 options文件中的配置

f = glob.glob(test_folder+'/*.png') # 获取所有的png图像文件
print(f)

for fl in f:
    mask = torch.zeros(opt.fineSize, opt.fineSize)   #指定大小的全0 torch
    if opt.mask_sub_type == 'fractal':         #mask的随机类型是fractal
        assert 1==2, "It is broken now..."
        mask = util.create_walking_mask()    # 创建初始的随机mask  调用util中的函数

    elif opt.mask_sub_type == 'rect':     #创建矩形的mask
        mask, rand_t, rand_l = util.create_rand_mask(opt)

    elif opt.mask_sub_type == 'island':   # #创建岛形的mask
        mask = util.wrapper_gmask(opt)
    
    print('Generating mask for test image: '+os.path.basename(fl))
    #保存mask到指定的目录
    util.save_image(mask.squeeze().numpy()*255, os.path.join(mask_folder, os.path.splitext(os.path.basename(fl))[0]+'_mask.png'))



