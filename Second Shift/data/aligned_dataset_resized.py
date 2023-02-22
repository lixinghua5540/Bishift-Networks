#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


#返回resize过的aligned的图像数据，所有图像都是统一尺寸
class AlignedDatasetResized(BaseDataset):
    def initialize(self, opt):
        self.opt = opt   #获取设置的参数
        self.root = opt.dataroot  #图像目录
        self.dir_A = opt.dataroot # More Flexible for users

        self.A_paths = sorted(make_dataset(self.dir_A))  #排序

        assert(opt.resize_or_crop == 'resize_and_crop')  #确定参数是resize_and_crop

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)  #将操作合并

    def __getitem__(self, index):
        A_path = self.A_paths[index]   #获取指定对应的图像路径
        A = Image.open(A_path).convert('RGB')  #打开图像并转化为RGB格式

        A = A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC) #对图像resize

        A = self.transform(A)##转化为tensor 并标准化

        #if (not self.opt.no_flip) and random.random() < 0.5:
        #    idx = [i for i in range(A.size(2) - 1, -1, -1)] # size(2)-1, size(2)-2, ... , 0
        #    idx = torch.LongTensor(idx)
        #    A = A.index_select(2, idx)

        # let B directly equals A
        B = A.clone()
        return {'A': A, 'B': B,
                'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)   #图像个数

    def name(self):
        return 'AlignedDatasetResized'
