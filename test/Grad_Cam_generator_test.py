"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os

import matplotlib.pyplot as plt
import numpy
import torchvision
from PIL import Image
import numpy as np
import torch
from torchsummary import summary
from torchvision import models, transforms
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms.functional as TF

myTransforms = transforms.Compose([
    transforms.ToTensor()])
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for layer, module in self.model._modules.items():
            x = module(x)  # Forward
            if layer == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
                return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # print("conv_output shape:", conv_output.shape, model_output.shape)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # print("guided_gradients shape: ",guided_gradients.shape)
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # print("target shape: ",target.shape)
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # print(np.max(weights), np.min(weights))
        # print("weights shape: ",weights.shape)
        # print("weights sort: ", weights.argsort()[::-1])
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # print("cam shape: ",cam.shape)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # print(np.max(cam), np.min(cam))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((32, 32), Image.LANCZOS))/255
        return cam


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    variance = 0
    mean = 0
    start_point2 = (12, 12)
    crop_size = (20, 20)
    padding = (0, 0, 12, 12)

    Resnet50 = torchvision.models.resnet50()
    inchannel = Resnet50.fc.in_features
    Resnet50.fc = nn.Linear(inchannel, 10)
    Resnet50.load_state_dict(torch.load('./model/task_224.pth.tar')['state_dict'])
    # print(Resnet50)
    #summary(model, input_size=(3, 256, 256))
    Resnet50.eval()
    Grad_Transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224)])

    train_dataset = torchvision.datasets.CIFAR10(root='/home/****/cifar/', train=True,
                                                 download=True,
                                                 transform=myTransforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    GradCam_numpy = np.zeros((50000, 32, 32))
    for i, (x, lables) in enumerate(train_loader):
        print(i)
        x_input = x
        y_input = TF.crop(x, start_point2[0], start_point2[1], crop_size[0], crop_size[0])
        y_input = TF.pad(y_input, padding)
        plt.figure(0)
        plt.axis('off')
        plt.imshow(np.transpose(x_input.squeeze().numpy(), (1, 2, 0)))
        plt.figure(1)
        plt.axis('off')
        plt.imshow(np.transpose(y_input.squeeze().numpy(), (1, 2, 0)))

        # crop_image_x = Grad_Transform(x_input)
        crop_image_y = Grad_Transform(x_input)

        lables = lables.item()
        target_class = lables

        # predict_x = Resnet50(crop_image_x)
        # predict_y = Resnet50(crop_image_y)
        # # print(y.data)
        # print('labels', 'predict_x', 'predict_y')
        # print(lables, predict_x.data.numpy().argsort()[0][-1], predict_y.data.numpy().argsort()[0][-1])

        layer_name = 'layer4'
        file_name_to_export = 'cat23'

        # Grad cam
        grad_cam = GradCam(Resnet50, layer_name)
        # Generate cam mask
        # cam_x = grad_cam.generate_cam(crop_image_x, target_class)
        cam_y = grad_cam.generate_cam(crop_image_y, target_class)
        # variance += np.std(cam_y)
        # mean += np.mean(cam_y)
        # print(np.mean(cam_y))
        GradCam_numpy[i, :, :] = cam_y
        # if i == 999 :
        #     break
        # plt.figure(2)
        # plt.imshow(cam_x)
        plt.figure(3)
        plt.imshow(cam_y)
        plt.axis('off')
        plt.tight_layout(pad=0)
        if i == 0 :
            plt.savefig("GRADCAM.svg", format='svg', dpi=150)
        plt.show()
    # Save mask
    # save_class_activation_images(original_image, cam, file_name_to_export)
    # save_class_activation_images(image_np, cam, file_name_to_export)
    # np.save('test_gradcam.npz', GradCam_numpy)
    # print('variance = ' + str(variance/1000))
    # print('mean = ' + str(mean/1000))
