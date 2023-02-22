#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


#AlignedDataset   对齐处理过的数据？
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt   #获取设置的参数
        self.dir_A = opt.dataroot    #图像目录
        self.A_paths = sorted(make_dataset(self.dir_A)) #排序
        if self.opt.offline_loading_mask:  #离线加载mask（即预先保存的mask）
            self.mask_folder = self.opt.training_mask_folder if self.opt.isTrain else self.opt.testing_mask_folder #mask目录
            self.mask_paths = sorted(make_dataset(self.mask_folder))  #排序

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]    #获取目录
        A = Image.open(A_path).convert('RGB')  #打开图像
        w, h = A.size

        #以下包括了align 图像的操作
        if w < h:   #宽小于高
            ht_1 = self.opt.loadSize * h // w
            wd_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)
        else:
            wd_1 = self.opt.loadSize * w // h
            ht_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)

        A = self.transform(A)
        h = A.size(1)
        w = A.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)] # size(2)-1, size(2)-2, ... , 0
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)

        # let B directly equals A
        B = A.clone()

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A.clone().zero_()
        if self.opt.offline_loading_mask:
            mask = Image.open(self.mask_paths[random.randint(0, len(self.mask_paths)-1)])
            mask = mask.resize((self.opt.fineSize, self.opt.fineSize), Image.NEAREST)
            mask = transforms.ToTensor()(mask)
        
        return {'A': A, 'B': B, 'M': mask,
                'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
