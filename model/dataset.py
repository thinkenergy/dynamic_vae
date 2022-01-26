# -*- coding: utf-8 -*-
# @Time : 2021/10/9 14:53
# @Author : huangshaobo,liujiachang,zhangyang 
# @Email : sdk.eval@thinkenergy.net.cn
# @File : dataset.py

import os
import torch

class Dataset:
    def __init__(self, data_path):
      
        self.data_path = data_path
        self.battery_dataset = []
        self.data_lst = os.listdir(data_path)
        for i in range(len(self.data_lst)):
            single_path=os.path.join(self.data_path, self.data_lst[i])
            train1 =torch.load(single_path)
            self.battery_dataset.append(train1)
    def __len__(self):
        return len(self.battery_dataset)

    def __getitem__(self, idx):
        file = self.battery_dataset[idx]
        return file
