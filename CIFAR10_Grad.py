# coding:utf-8
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.transforms.functional as TF

start_point2 = (12, 12)
crop_size = (20, 20)
padding = (0, 0, 12, 12)
class CIFAR10_Gradcam(CIFAR10):
    def __init__(self, root1, root2, train = True, download=False,transform=None):
        super(CIFAR10_Gradcam, self).__init__(root1, train=train, download=download, transform=transform)
        self.gradcam_path = root2
        self.train = train
        self.train_numpy = np.load(self.gradcam_path + '/train/train_gradcam.npz.npy')
        self.test_numpy = np.load(self.gradcam_path + '/test/test_gradcam.npz.npy')
    def __getitem__(self, index):
        image, labels = super(CIFAR10_Gradcam, self).__getitem__(index)
        if self.train:
            gradcam = torch.tensor(self.train_numpy[index, :, :], dtype=torch.float).unsqueeze(0)
        if bool(1-self.train):
            gradcam = torch.tensor(self.test_numpy[index, :, :], dtype=torch.float).unsqueeze(0)
        x_input = image
        y_input = TF.crop(image, start_point2[0], start_point2[1], crop_size[0], crop_size[0])
        y_input = TF.pad(y_input, padding)
        side_info = y_input

        return x_input, side_info, gradcam, labels

