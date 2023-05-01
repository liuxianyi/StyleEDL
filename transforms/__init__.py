from .multiscale_crop import MultiScaleCrop
from .normalize import Normalize

from torchvision import transforms


def get_transforms(image_size, mode, dataset, isNormalize=True):
    transforms_list = []

    # resize
    if mode == 'train':
        transforms_list.append(
            transforms.Resize((image_size + 64, image_size + 64)))
        transforms_list.append(
            MultiScaleCrop(image_size,
                           scales=(1.0, 0.875, 0.75, 0.66, 0.5),
                           max_distort=2))
    else:
        transforms_list.append(transforms.Resize((image_size, image_size)))

    transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # ToTensor
    transforms_list.append(transforms.ToTensor())

    # normalize
    if isNormalize:
        transforms_list.append(Normalize(dataset))

    return transforms.Compose(transforms_list)
