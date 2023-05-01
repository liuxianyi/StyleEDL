# -*- encoding: utf-8 -*-

import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset_LDL(Dataset):
    def __init__(
        self,
        root_path,
        mode,
        transforms,
    ):
        super().__init__()
        self.transforms = transforms
        self.images_path = []
        self.labels = []
        self.cls = []

        if mode == "train":
            file_name = 'ground_truth_train.txt'
        elif mode == "test":
            file_name = 'ground_truth_test.txt'

        # 读入文件
        with open(os.path.join(root_path, file_name), 'r') as f:
            file = f.readlines()

        for line in file:
            temp = line.rstrip('\n').rstrip(' ').split(' ')
            self.images_path.append(os.path.join(root_path, "images", temp[0]))
            label = [eval(i) for i in temp[1:-1]]
            self.cls.append(eval(temp[-1]))
            self.labels.append(label)

    def __getitem__(self, index):
        original_img = Image.open(self.images_path[index]).convert('RGB')
        label = torch.FloatTensor(self.labels[index])
        cls = self.cls[index]
        original_img = self.transforms(original_img)
        return original_img, label, cls

    def __len__(self) -> int:
        return len(self.images_path)
