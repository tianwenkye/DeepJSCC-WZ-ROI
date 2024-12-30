import math
import numpy as np
import torch.nn as nn
import torch
import os
from GDN import GDN

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding ,bias=False)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = GDN(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding = 0):
        super(deconv_block, self).__init__()
        self.deconv = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  output_padding = output_padding)
        self.bn = GDN(out_channels, inverse=True)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, activate_func='prelu'):
        out = self.deconv(x)
        out = self.bn(out)
        if activate_func =='prelu':
            out = self.prelu(out)
        elif activate_func =='sigmoid':
            out = self.sigmoid(out)
        return out

# class AF_block(nn.Module):
#     def __init__(self, Nin, Nh, No):
#         super(AF_block, self).__init__()
#         self.fc1 = nn.Linear(Nin+1, Nh)
#         self.fc2 = nn.Linear(Nh, No)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x, snr):
#         # out = F.adaptive_avg_pool2d(x, (1,1))
#         # out = torch.squeeze(out)
#         # out = torch.cat((out, snr), 1)
#         if snr.shape[0]>1:
#             snr = snr.squeeze(1)
#             snr = snr.unsqueeze(1)
#         mu = torch.mean(x, (2, 3))
#         out = torch.cat((mu, snr), 1)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         out = out.unsqueeze(2)
#         out = out.unsqueeze(3)
#         out = out*x
#         return out

        
# The Encoder model with attention feature blocks
class Encoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(Encoder, self).__init__()    
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv//2
        padding_L = (kernel_sz-1)//2
        self.conv1 = conv_block(3, Nc_conv, kernel_size = kernel_sz, stride = 2, padding=padding_L)
        self.conv2 = conv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv3 = conv_block(Nc_conv, enc_N, kernel_size=kernel_sz, stride=1, padding=padding_L)

        self.flatten = nn.Flatten() 
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out

# The Decoder model with attention feature blocks
class Decoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_deconv):
        super(Decoder, self).__init__()
        self.enc_shape = enc_shape
        Nh_AF1 = enc_shape[0]//2
        Nh_AF = Nc_deconv//2
        padding_L = (kernel_sz-1)//2
        self.deconv1 = deconv_block(self.enc_shape[0], Nc_deconv, kernel_size = kernel_sz, stride = 1,  padding=padding_L)
        self.deconv2 = deconv_block(Nc_deconv, Nc_deconv, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding=1)
        self.deconv3 = deconv_block(Nc_deconv, 3, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)

    def forward(self, x):
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out, 'sigmoid')

        return out
class Decoder2(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_deconv):
        super(Decoder2, self).__init__()
        self.enc_shape = enc_shape
        Nh_AF1 = enc_shape[0]//2
        Nh_AF = Nc_deconv//2
        padding_L = (kernel_sz-1)//2
        self.deconv1 = deconv_block(self.enc_shape[0], Nc_deconv, kernel_size = kernel_sz, stride = 1,  padding=padding_L)
        self.deconv2 = deconv_block(Nc_deconv, Nc_deconv, kernel_size = kernel_sz, stride = 2,  padding=padding_L, output_padding=1)
        self.deconv3 = deconv_block(Nc_deconv, Nc_deconv, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)
        self.deconv4 = deconv_block(Nc_deconv, 3, kernel_size=kernel_sz, stride=1, padding=padding_L)

    def forward(self, x):
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out, 'sigmoid')

        return out
# Power normalization before transmission
# Note: if P = 1, the symbol power is 2
# If you want to set the average power as 1, please change P as P=1/np.sqrt(2)
def Power_norm(z, P = 1):
    batch_size, z_dim = z.shape
    z_power = torch.sqrt(torch.sum(z**2, 1))
    z_M = z_power.repeat(z_dim, 1)
    return np.sqrt(P*z_dim)*z/z_M.t()

def Power_norm_complex(z, P = 1): 
    batch_size, z_dim = z.shape
    z_com = torch.complex(z[:, 0:z_dim:2], z[:, 1:z_dim:2])
    z_com_conj = torch.complex(z[:, 0:z_dim:2], -z[:, 1:z_dim:2])
    z_power = torch.sum(z_com*z_com_conj, 1).real
    z_M = z_power.repeat(z_dim//2, 1)
    z_nlz = np.sqrt(P*z_dim)*z_com/torch.sqrt(z_M.t())
    z_out = torch.zeros(batch_size, z_dim).cuda()
    z_out[:, 0:z_dim:2] = z_nlz.real
    z_out[:, 1:z_dim:2] = z_nlz.imag
    return z_out

# The (real) AWGN channel    
def AWGN_channel(x, snr, P = 1):
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(P/gamma)*torch.randn(batch_size, length).cuda()
    y = x+noise
    return y

def AWGN_complex(x, snr, Ps = 1):  
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    n_I = torch.sqrt(Ps/gamma)*torch.randn(batch_size, length).cuda()
    n_R = torch.sqrt(Ps/gamma)*torch.randn(batch_size, length).cuda()
    noise = torch.complex(n_I, n_R)
    y = x + noise
    return y

# Please set the symbol power if it is not a default value
def Fading_channel(x, snr, P = 1):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    h_I = torch.randn(batch_size, K).cuda()
    h_R = torch.randn(batch_size, K).cuda() 
    h_com = np.sqrt(1/2)*torch.complex(h_I, h_R)
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_com = h_com*x_com
    
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    noise = torch.complex(n_I, n_R)
    
    y_add = y_com + noise
    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).cuda()
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out



# Note: if P = 1, the symbol power is 2
# If you want to set the average power as 1, please change P as P=1/np.sqrt(2)
def Power_norm_VLC(z, cr, P = 1):
    batch_size, z_dim = z.shape
    Kv = torch.ceil(z_dim*cr).int()
    z_power = torch.sqrt(torch.sum(z**2, 1))
    z_M = z_power.repeat(z_dim, 1).cuda()
    return torch.sqrt(Kv*P)*z/z_M.t()


def AWGN_channel_VLC(x, snr, cr, P = 1):
    batch_size, length = x.shape
    gamma = 10 ** (snr / 10.0)
    mask = mask_gen(length, cr).cuda()
    noise = torch.sqrt(P/gamma)*torch.randn(batch_size, length).cuda()
    noise = noise*mask
    y = x+noise
    return y


def Fading_channel_VLC(x, snr, cr, P = 1):
    gamma = 10 ** (snr / 10.0)
    [batch_size, feature_length] = x.shape
    K = feature_length//2
    
    mask = mask_gen(K, cr).cuda()
    h_I = torch.randn(batch_size, K).cuda()
    h_R = torch.randn(batch_size, K).cuda() 
    h_com = np.sqrt(1/2)*torch.complex(h_I, h_R)
    x_com = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_com = h_com*x_com
    
    n_I = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    n_R = torch.sqrt(P/gamma)*torch.randn(batch_size, K).cuda()
    noise = torch.complex(n_I, n_R)*mask
    
    y_add = y_com + noise
    y = y_add/h_com
    
    y_out = torch.zeros(batch_size, feature_length).cuda()
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out


def Channel(z, snr, channel_type = 'AWGN'):
    z = Power_norm(z)
    if channel_type == 'AWGN':
        z = AWGN_channel(z, snr)
    elif channel_type == 'Fading':
        z = Fading_channel(z, snr)
    return z


def Channel_VLC(z, snr, cr, channel_type = 'AWGN'):
    z = Power_norm_VLC(z, cr)
    if channel_type == 'AWGN':
        z = AWGN_channel_VLC(z, snr, cr)
    elif channel_type == 'Fading':
        z = Fading_channel_VLC(z, snr, cr)
    return z


def mask_gen(N, cr, ch_max = 48):
    MASK = torch.zeros(cr.shape[0], N).int()
    nc = N//ch_max
    for i in range(0, cr.shape[0]):
        L_i = nc*torch.round(ch_max*cr[i]).int()
        MASK[i, 0:L_i] = 1
    return MASK

# The DeepJSCC model

class DeepJSCC(nn.Module):
    def __init__(self, enc_shape, Kernel_sz, Nc):
        super(DeepJSCC, self).__init__()
        self.encoder = Encoder(enc_shape, Kernel_sz, Nc)
        self.decoder = Decoder(enc_shape, Kernel_sz, Nc)
    def forward(self, x, snr, channel_type = 'AWGN'):
        z = self.encoder(x)
        z = Channel(z, snr, channel_type)
        out = self.decoder(z)
        return out
class DeepJSCC2(nn.Module):
    def __init__(self, enc_shape, Kernel_sz, Nc):
        super(DeepJSCC2, self).__init__()
        self.encoder = Encoder(enc_shape, Kernel_sz, Nc)
        self.decoder = Decoder2(enc_shape, Kernel_sz, Nc)
    def forward(self, x, snr, channel_type = 'AWGN'):
        z = self.encoder(x)
        z = Channel(z, snr, channel_type)
        out = self.decoder(z)
        return out



class DistributedAutoEncoder(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(DistributedAutoEncoder, self).__init__()
        self.enc_shape = enc_shape
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv // 2
        padding_L = (kernel_sz - 1) // 2
        ####################################encoder
        self.conv1 = conv_block(3, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv2 = conv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv3 = conv_block(Nc_conv, enc_N, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.flatten = nn.Flatten()
        ####################################decoder
        self.deconv1 = deconv_block(self.enc_shape[0]*2, Nc_conv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv2 = deconv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        self.deconv3 = deconv_block(Nc_conv*2, 3, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out

    def encode_sm(self, x):
        s1 = self.conv1(x)
        s2 = self.conv2(s1)
        s2 = self.conv3(s2)
        s2 = self.flatten(s2)
        return s1, s2
    def decode_sm(self, x, w):
        s1, s2 = self.encode_sm(w)
        s2 = s2.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = torch.cat((out, s2), 1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = torch.cat((out, s1), 1)
        out = self.deconv3(out,)

        return out
    def forward(self, x, y, snr, channel_type):
        z = self.encode(x)
        z = Channel(z, snr, channel_type)
        out = self.decode_sm(z, y)
        return out

class DistributedAutoEncoder2(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(DistributedAutoEncoder2, self).__init__()
        self.enc_shape = enc_shape
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv // 2
        padding_L = (kernel_sz - 1) // 2
        ####################################encoder
        self.conv1 = conv_block(3, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv2 = conv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv3 = conv_block(Nc_conv, enc_N, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.flatten = nn.Flatten()
        ####################################decoder
        self.deconv1 = deconv_block(self.enc_shape[0]*2, Nc_conv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv2 = deconv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        self.deconv3 = deconv_block(Nc_conv*2, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)
        self.deconv4 = deconv_block(Nc_conv +3, 3, kernel_size=kernel_sz, stride=1, padding=padding_L)

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out

    def encode_sm(self, x):
        s1 = self.conv1(x)
        s2 = self.conv2(s1)
        s2 = self.conv3(s2)
        s2 = self.flatten(s2)
        return s1, s2
    def decode_sm(self, x, w):
        s1, s2 = self.encode_sm(w)
        s2 = s2.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = torch.cat((out, s2), 1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = torch.cat((out, s1), 1)
        out = self.deconv3(out)
        out = torch.cat((out, w), 1)
        out = self.deconv4(out, 'sigmoid')

        return out
    def forward(self, x, y, snr, channel_type):
        z = self.encode(x)
        z = Channel(z, snr, channel_type)
        out = self.decode_sm(z, y)
        return out
class DistributedAutoEncoder3(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(DistributedAutoEncoder3, self).__init__()
        self.enc_shape = enc_shape
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv // 2
        padding_L = (kernel_sz - 1) // 2
        ####################################encoder
        self.conv1 = conv_block(3, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv2 = conv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv3 = conv_block(Nc_conv, enc_N, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.flatten = nn.Flatten()
        ####################################decoder
        self.deconv1 = deconv_block(self.enc_shape[0], Nc_conv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv2 = deconv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        self.deconv3 = deconv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L, output_padding=1)
        self.deconv4 = deconv_block(Nc_conv +3, 3, kernel_size=kernel_sz, stride=1, padding=padding_L)

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out

    def encode_sm(self, x):
        s1 = self.conv1(x)
        s2 = self.conv2(s1)
        s2 = self.conv3(s2)
        s2 = self.flatten(s2)
        return s1, s2
    def decode_sm(self, x, w):
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = torch.cat((out, w), 1)
        out = self.deconv4(out, 'sigmoid')

        return out
    def forward(self, x, y, snr, channel_type):
        z = self.encode(x)
        z = Channel(z, snr, channel_type)
        out = self.decode_sm(z, y)
        return out


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=3, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out



class DDJSCC_Grad(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(DDJSCC_Grad, self).__init__()
        self.enc_shape = enc_shape
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv // 2
        padding_L = (kernel_sz - 1) // 2
        ####################################encoder
        self.conv1 = conv_block(3, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv2 = conv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv3 = conv_block(Nc_conv, enc_N, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.flatten = nn.Flatten()
        ####################################decoder
        self.deconv1 = deconv_block(self.enc_shape[0] * 2, Nc_conv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv2 = deconv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        self.deconv3 = deconv_block(Nc_conv * 2, 3, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        ####################################Conditional_Network
        self.condition_1 = nn.Sequential(
            conv(4, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            conv(64, 64, 3, 2, 1)
        )
        self.condition_2 = nn.Sequential(
            conv(4, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            conv(64, 64, 3, 1, 1)
        )
        self.sft_1 = SFT(256, 64)
        self.sft_2 = SFT(512, 64)

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out

    def encode_sm(self, x):
        s1 = self.conv1(x)
        s2 = self.conv2(s1)
        s2 = self.conv3(s2)
        s2 = self.flatten(s2)
        return s1, s2
    def decode_sm(self, x, w, gradcam):
        condition = torch.cat((w, gradcam), 1)
        condition1 = self.condition_1(condition)
        condition2 = self.condition_2(condition)

        s1, s2 = self.encode_sm(w)
        s2 = s2.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = torch.cat((out, s2), 1)
        out = self.deconv1(out)
        out = self.sft_1(out, condition1)
        out = self.deconv2(out)
        out = torch.cat((out, s1), 1)
        out = self.sft_2(out, condition2)
        out = self.deconv3(out, 'sigmoid')

        return out
    def forward(self, x, y, gradcam, snr, channel_type):
        z = self.encode(x)
        z = Channel(z, snr, channel_type)
        out = self.decode_sm(z, y, gradcam)
        return out

class DDJSCC_Grad_ablation(nn.Module):
    def __init__(self, enc_shape, kernel_sz, Nc_conv):
        super(DDJSCC_Grad_ablation, self).__init__()
        self.enc_shape = enc_shape
        enc_N = enc_shape[0]
        Nh_AF = Nc_conv // 2
        padding_L = (kernel_sz - 1) // 2
        ####################################encoder
        self.conv1 = conv_block(3, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv2 = conv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L)
        self.conv3 = conv_block(Nc_conv, enc_N, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.flatten = nn.Flatten()
        ####################################decoder
        self.deconv1 = deconv_block(self.enc_shape[0] * 2, Nc_conv, kernel_size=kernel_sz, stride=1, padding=padding_L)
        self.deconv2 = deconv_block(Nc_conv, Nc_conv, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        self.deconv3 = deconv_block(Nc_conv * 2, 3, kernel_size=kernel_sz, stride=2, padding=padding_L,
                                    output_padding=1)
        ####################################Conditional_Network
        self.condition_1 = nn.Sequential(
            conv(3, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            conv(64, 64, 3, 2, 1)
        )
        self.condition_2 = nn.Sequential(
            conv(3, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, True),
            conv(64, 64, 3, 1, 1)
        )
        self.sft_1 = SFT(256, 64)
        self.sft_2 = SFT(512, 64)

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out

    def encode_sm(self, x):
        s1 = self.conv1(x)
        s2 = self.conv2(s1)
        s2 = self.conv3(s2)
        s2 = self.flatten(s2)
        return s1, s2
    def decode_sm(self, x, w):
        condition1 = self.condition_1(w)
        condition2 = self.condition_2(w)

        s1, s2 = self.encode_sm(w)
        s2 = s2.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = x.view(-1, self.enc_shape[0], self.enc_shape[1], self.enc_shape[2])
        out = torch.cat((out, s2), 1)
        out = self.deconv1(out)
        out = self.sft_1(out, condition1)
        out = self.deconv2(out)
        out = torch.cat((out, s1), 1)
        out = self.sft_2(out, condition2)
        out = self.deconv3(out, 'sigmoid')

        return out
    def forward(self, x, y, snr, channel_type):
        z = self.encode(x)
        z = Channel(z, snr, channel_type)
        out = self.decode_sm(z, y)
        return out

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    enc_out_shape = [6, 32//4, 32 // 4]
    net = DDJSCC_Grad_ablation(enc_out_shape, 3, 256).cuda()
    x_input = torch.randn(64, 3, 32, 32).cuda()
    y_input = torch.randn(64, 3, 32, 32).cuda()
    gradcam = torch.randn(64, 1, 32, 32).cuda()
    SNR_TRAIN = torch.tensor(5).cuda()

    print(net(x_input, y_input, SNR_TRAIN, 'AWGN')[0].shape)
















