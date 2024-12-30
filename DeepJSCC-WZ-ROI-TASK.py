import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
from models import DistributedAutoEncoder, DDJSCC_Grad
from CIFAR10_Grad import CIFAR10_Gradcam
from resnet_model import *
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total



os.environ['CUDA_VISIBLE_DEVICES'] = '1'
myTransforms = transforms.Compose([
    transforms.ToTensor()])


#  load
train_dataset = CIFAR10_Gradcam(root1='/home/****/cifar/', root2='/home/****/gradcam_crop',train=True, download=True,
                                             transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = CIFAR10_Gradcam(root1='/home/****/cifar/', root2='/home/****/gradcam_crop',train=False, download=True,
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
bestLoss = 0.1
current_epoch = 0
CONTINUE_TRAINING = True
LOAD_PRETRAIN = False
cr = 6
snr = 10.0
enc_out_shape = [cr, 32//4, 32//4]
KSZ = str(Kernel_sz ) +'x ' +str(Kernel_sz ) +'_'
model = DDJSCC_Grad(enc_out_shape, Kernel_sz, N_channels).cuda()
criterion1 = nn.MSELoss().cuda()
criterion2 = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
########################################################################
if CONTINUE_TRAINING == True:
    model.load_state_dict(torch.load('./JSCC_models/DDJSCC_gradcam_' + str(cr) + '_.pth.tar')['state_dict'])
    current_epoch = 0
for epoch in range(current_epoch, EPOCHS):
    model.train()
    Resnet50.eval()
    print('========================')
    print('lr:%.4e ' % optimizer.param_groups[0]['lr'])
    # if epoch == 40:
    #     optimizer.param_groups[0]['lr'] =  0.5*1e-4

    # Model training
    for i,(x, y, gradcam, labels) in enumerate(train_loader):
        # print(i)%
        x = x.cuda()
        y = y.cuda()
        gradcam = gradcam.cuda()
        labels = labels.cuda()

        SNR_TRAIN = torch.tensor(5).cuda()
        # 经过联合编解码器
        x_rec = model(x, y, gradcam, SNR_TRAIN, CHANNEL)
        #loss1
        loss1 = criterion1(x, x_rec)
        loss1 = loss1.mean()
        #下游任务
        predict_label = Resnet50.forward(x_rec)
        # loss2
        loss2 = criterion2(predict_label, labels)
        loss = loss1 + 0.001*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % PRINT_RREQ == 0:
            print \
                ('Epoch: [{0}][{1}/{2}]\t' 'Loss1 {loss1:.4f}\t' 'Loss2 {loss2:.4f}\t''Loss {loss:.4f}\t'.format(epoch,
                                                                                                                 i,
                                                                                                                 len(train_loader),
                                                                                                                 loss1=loss1.item(),
                                                                                                                 loss2=loss2.item(),
                                                                                                                 loss=loss.item()))

    # Model Evaluation
    model.eval()
    totalLoss = 0
    total_acc = 0
    with torch.no_grad():
        for i, (test_x, test_y, test_gradcam, test_labels) in enumerate(test_loader):
            test_x = test_x.cuda()
            test_y = test_y.cuda()
            test_gradcam = test_gradcam.cuda()
            test_labels = test_labels.cuda()

            SNR_TEST = torch.tensor(5).cuda()
            x_rec = model(test_x, test_y, test_gradcam, SNR_TEST, CHANNEL)
            # loss1
            loss1 = criterion1(test_x, x_rec)
            loss1 = loss1.mean()
            # 下游任务
            predict_label = Resnet50.forward(x_rec)
            total_acc += get_acc(predict_label, test_labels)* test_labels.size(0)
            totalLoss += loss1.item() * test_labels.size(0)

        averageLoss = totalLoss / (len(test_dataset))
        averageacc = total_acc /(len(test_dataset))
        print('averageLoss=', averageLoss)
        print('averageacc=', averageacc)
        if averageacc > bestLoss:
            # Model saving
            if not os.path.exists('./JSCC_models'):
                os.makedirs('./JSCC_models')
            torch.save({'state_dict': model.state_dict(), }, './JSCC_models/DDJSCC_grad_task_'+str(cr)+'_.pth.tar')
            print('Model saved')
            bestLoss = averageacc