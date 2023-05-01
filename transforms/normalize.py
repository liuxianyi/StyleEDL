# -*- encoding: utf-8 -*-
from torchvision import transforms


class Normalize(object):
    def __init__(self, dataset) -> None:
        super().__init__()
        # train
        if dataset == "Emotion6":
            self.means = [0.41779748, 0.38421513, 0.34800839]
            self.stdevs = [0.23552664, 0.22541416, 0.21950753]
        elif dataset == 'Flickr_LDL':
            self.means = [0.43735039, 0.39944456, 0.36520021]
            self.stdevs = [0.24785846, 0.23636487, 0.23396503]
        elif dataset == 'Twitter_LDL':
            self.means = [0.49303343, 0.4541828, 0.43356296]
            self.stdevs = [0.25708641, 0.2484328, 0.24492859]
        elif dataset == 'general':
            self.means = [0.5, 0.5, 0.5]
            self.stdevs = [0.5, 0.5, 0.5]
        else:
            self.means = [0.49276434, 0.45391981, 0.43331505]
            self.stdevs = [0.25703167, 0.24834259, 0.24485385]

    def __call__(self, image):
        return transforms.Normalize(self.means, self.stdevs)(image)

    def __str__(self) -> str:
        return self.__class__.__name__
