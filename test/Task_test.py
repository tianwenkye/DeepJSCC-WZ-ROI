# coding:utf-8
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import os
from models import DistributedAutoEncoder
import torchvision.transforms.functional as TF
from resnet_model import *
def PSNR(loss):
    return 10 * math.log10(1/loss)
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


start_point2 = (12, 12)
crop_size = (20, 20)
padding = (0, 0, 12, 12)


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
myTransforms = transforms.Compose([
    transforms.ToTensor()])


#  load
train_dataset = torchvision.datasets.CIFAR10(root='/home/****/cifar/', train=True, download=True,
                                             transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='/home/****/cifar/', train=False, download=True,
                                            transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 定义模型
Resnet50 = ResNet50()
Resnet50.load_state_dict(torch.load('./model/resnet50_v2.pth.tar')['state_dict'])
# 损失函数及优化器
# GPU加速
Resnet50 = Resnet50.cuda()
###################################################
EPOCHS = 100
LEARNING_RATE = 1e-4
PRINT_RREQ = 200
SAVE_RREQ = 500
CHANNEL = 'AWGN'  # Choose AWGN or Fading
N_channels = 256
Kernel_sz = 5
bestLoss = 1000
current_epoch = 0
CONTINUE_TRAINING = True
LOAD_PRETRAIN = False
cr = 2
snr = 10.0
enc_out_shape = [cr, 32//4, 32//4]
KSZ = str(Kernel_sz ) +'x ' +str(Kernel_sz ) +'_'
# model = DeepJSCC(enc_out_shape, Kernel_sz, N_channels).cuda()
model = DistributedAutoEncoder(enc_out_shape, Kernel_sz, N_channels).cuda()
model.load_state_dict(torch.load('./JSCC_models/DDJSCC_task_' + str(cr) + '_.pth.tar')['state_dict'])

criterion1 = nn.MSELoss().cuda()
criterion2 = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
########################################################################
PSNR_ave = np.zeros((10, 10))
ACC_ave = np.zeros((10, 10))
for m in range(0, 10):
    for k in range(0, 10):
        print('Evaluating DeepJSCC with CR = ' + str(cr) + ' and SNR = ' + str(k ) + 'dB')
        totalLoss = 0
        model.eval()
        total_acc = 0


        Resnet50.eval()
        with torch.no_grad():
            for i, (image_0, test_label) in enumerate(test_loader):
                x_input = image_0
                y_input = TF.crop(image_0, start_point2[0], start_point2[1], crop_size[0], crop_size[0])
                y_input = TF.pad(y_input, padding)
                x_input = x_input.cuda()
                y_input = y_input.cuda()
                SNR = torch.tensor(k).cuda()
                test_label = test_label.cuda()
                test_rec = model(x_input, y_input, SNR, CHANNEL)
                predict_label = Resnet50.forward(x_input)

                total_acc += get_acc(predict_label, test_label) * test_label.size(0)
                totalLoss += criterion1(test_rec, x_input).item() * x_input.size(0)
            averageLoss = totalLoss / (len(test_dataset))
            averagePSNR = PSNR(averageLoss)
            averageacc = total_acc / (len(test_dataset))
            print('PSNR = ' + str(averagePSNR))
            print('averageacc=', averageacc)

        ACC_ave[m, k] = averageacc
        PSNR_ave[m, k] = averagePSNR
PSNR_mean = np.mean(PSNR_ave, 0)
ACC_mean = np.mean(ACC_ave, 0)
print(PSNR_mean)
print(ACC_mean)