import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
from models import DistributedAutoEncoder
import torchvision.transforms.functional as TF

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total



os.environ['CUDA_VISIBLE_DEVICES'] = '1'
myTransforms = transforms.Compose([
    transforms.ToTensor()])
start_point2 = (12, 12)
crop_size = (20, 20)
padding = (0, 0, 12, 12)

#  load
train_dataset = torchvision.datasets.CIFAR10(root='/home/****/cifar/', train=True, download=True,
                                             transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='/home/****/cifar/', train=False, download=True,
                                            transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

EPOCHS = 100
LEARNING_RATE = 1e-4
PRINT_RREQ = 200
SAVE_RREQ = 500
CHANNEL = 'AWGN'  # Choose AWGN or Fading
N_channels = 256
Kernel_sz = 5
bestLoss = 1000
current_epoch = 0
CONTINUE_TRAINING = False
LOAD_PRETRAIN = False
cr = 6
enc_out_shape = [cr, 32//4, 32//4]
KSZ = str(Kernel_sz ) +'x ' +str(Kernel_sz ) +'_'
model = DistributedAutoEncoder(enc_out_shape, Kernel_sz, N_channels).cuda()
criterion1 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
########################################################################
for epoch in range(current_epoch, EPOCHS):
    model.train()
    # Resnet50.eval()
    print('========================')
    print('lr:%.4e ' % optimizer.param_groups[0]['lr'])
    # if epoch == 40:
    #     optimizer.param_groups[0]['lr'] =  0.5*1e-4

    # Model training
    for i,(x, _) in enumerate(train_loader):
        # print(i)%
        x_input = x
        y_input = TF.crop(x, start_point2[0], start_point2[1], crop_size[0], crop_size[0])
        y_input = TF.pad(y_input, padding)
        x_input = x_input.cuda()
        y_input = y_input.cuda()
        x = x.cuda()

        SNR_TRAIN = torch.tensor(5).cuda()
        #经过联合编解码器
        x_rec = model(x_input, y_input, SNR_TRAIN, CHANNEL)
        #loss1
        loss1 = criterion1(x_input, x_rec)
        loss1 = loss1.mean()
        #下游任务
        # predict_label = Resnet50.forward(x_rec)
        #loss2
        # loss2 = criterion2(predict_label, label)
        # loss = loss1 + 0.01*loss2
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        if i % PRINT_RREQ == 0:
            print \
                ('Epoch: [{0}][{1}/{2}]\t' 'Loss1 {loss1:.4f}\t' .format(epoch, i, len(train_loader), loss1=loss1.item()))

   # Model Evaluation
    model.eval()
    totalLoss = 0
    total_acc = 0
    with torch.no_grad():
        for i, (test_input, _) in enumerate(test_loader):
            x_input = test_input
            y_input = TF.crop(test_input, start_point2[0], start_point2[1], crop_size[0], crop_size[0])
            y_input = TF.pad(y_input, padding)
            x_input = x_input.cuda()
            y_input = y_input.cuda()
            SNR_TEST = torch.tensor(5).cuda()
            x_rec = model(x_input, y_input, SNR_TEST, CHANNEL)
            # loss1
            loss1 = criterion1(x_input, x_rec)
            loss1 = loss1.mean()
            totalLoss += loss1.item() * test_input.size(0)

        averageLoss = totalLoss / (len(test_dataset))
        print('averageLoss=', averageLoss)
        if averageLoss < bestLoss:
            # Model saving
            if not os.path.exists('./JSCC_models'):
                os.makedirs('./JSCC_models')
            torch.save({'state_dict': model.state_dict(), }, './JSCC_models/DDJSCC_cifar_'+str(cr)+'_.pth.tar')
            print('Model saved')
            bestLoss = averageLoss