# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:03:46 2022

@author: Yibin Ye
"""
import random

from torchvision import transforms
import torch
import torch.utils.data as data


class myTransform_poc_8(data.Dataset):
    """`myDataset
    Args:
        root (string): Root directory where images are.
        #transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version.
    """
    #3/24修改，加入了训练集判断和旋转，以实现图像增强
    def __init__(self, root, train=None):

        self.opt_mean = 0.35876667499542236
        self.opt_std = 0.3196282386779785
        self.sar_mean = 0.24362623691558838
        self.sar_std = 0.19221791625022888

        self.data_file = root
        self.train = train
        self.transform_train_opt = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.RandomHorizontalFlip(p=0.2),
                                                       transforms.RandomVerticalFlip(p=0.2),
                                                       transforms.GaussianBlur(3,sigma=(0.1, 2.0)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(self.opt_mean,), std=(self.opt_std,)),
                                                       ])
        self.transform_train_sar = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.RandomHorizontalFlip(p=0.2),
                                                       transforms.RandomVerticalFlip(p=0.2),
                                                       transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(self.sar_mean,),std=(self.sar_std,)),
                                                       ])  # 归一化
        self.transform_test_opt = transforms.Compose([transforms.ToPILImage(),# transforms类主要进行图像转化
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(self.opt_mean,), std=(self.opt_std,))])  # 归一化
        self.transform_test_sar = transforms.Compose([transforms.ToPILImage(),  # transforms类主要进行图像转化
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(self.sar_mean,), std=(self.sar_std,))])  # 归一化
        self.images, self.distances  = torch.load(self.data_file)


    def __len__(self):
        # return len(self.sar)
        return len(self.distances)


    def __getitem__(self, index):

        images, d = self.images[index], self.distances[index]
        if self.train:
            k = random.randint(1,1000000)
            torch.manual_seed(k)
            sar = self.transform_train_sar(images[0])
            opt_list = []
            for i,opt in enumerate(images[1:len(images)]):
                torch.manual_seed(k)
                opt_list.append(self.transform_train_opt(opt))
            return sar, opt_list, d
        else:
            sar = self.transform_test_sar(images[0])
            opt_list = []
            for i, opt in enumerate(images[1:len(images)]):
                opt_list.append(self.transform_test_opt(opt))
            return sar, opt_list, d

