# Credits: Michael Rebsamen
# https://github.com/SCAN-NRAD/DL-DiReCT/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, ReplicationPad2d, Dropout
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
import random

# model settings
DIM = 96
stack_depth = 5

def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm3d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            #("dropout", nn.Dropout(p=0.2))
    ]))
    return layer

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(dilation)),
            ("conv1", nn.Conv2d(in_channel, growth_rate, kernel_size=kernel_size, dilation = dilation,padding=0)),
            ("dropout", nn.Dropout(p=0.0))]))
    
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out
    
class AttentionModule(nn.Module):
    def __init__(self, in_channel , intermediate_channel, out_channel, kernel_size=3):
        super(AttentionModule,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, intermediate_channel, kernel_size=kernel_size,padding=0)),
            ("bn2", nn.InstanceNorm2d(intermediate_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(intermediate_channel, out_channel, kernel_size=kernel_size,padding=0)),
            ("sigmoid", nn.Sigmoid())]))
    
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = x * out
        return out
    
def center_crop(layer, target_size):
    _, _, layer_width, layer_height = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start:(start + target_size), start:(start + target_size)]
    return crop

def concatenate(link, layer):
    concat = torch.cat([link, layer], 1)
    return concat

def dense_atrous_bottleneck(in_channel, growth_rate = 12, depth = [4,4,4,4]):
    layer_dict = OrderedDict()
    for idx, growth_steps in enumerate(depth):
        dilation_rate = 2**idx
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}".format(dilation_rate,y)] = DilatedDenseUnit(in_channel, 
                                                                        growth_rate, 
                                                                        kernel_size=3, 
                                                                        dilation = dilation_rate)
            in_channel = in_channel + growth_rate
        
        layer_dict["attention_{}".format(dilation_rate)] = AttentionModule(in_channel, in_channel//4, in_channel)
        
    return nn.Sequential(layer_dict), in_channel

class UNET_3D_to_2D(nn.Module):
    def __init__(self, depth, channels_in = 1, 
                 channels_2d_to_3d = 32, channels = 32, output_channels = 1, slices=stack_depth, 
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D, self).__init__()
        self.main_modules = []
        
        self.depth = depth
        self.slices = slices
        
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])
        
        
        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i), 
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate, 
                                                                       depth = dilated_layers)
        self.main_modules.append(self.bottleneck)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)),
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)), 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth, -1, -1)])
        
        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        self.main_modules.append(self.last)
        
        self.logvar = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        # down
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out).clone()
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)
        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        
        out = self.bottleneck(out)
        
        links.reverse()

        # up
        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        #logvar = self.logvar(out)
        #logvar = -torch.exp(logvar)

        return torch.unsqueeze(pred, 1) #, logvar


# This is the heteroscedastic model
class UNET_3D_to_2D_hetero(nn.Module):
    def __init__(self, depth, channels_in = 1, 
                 channels_2d_to_3d = 32, channels = 32, output_channels = 1, slices=stack_depth, 
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D_hetero, self).__init__()
        self.main_modules = []
        
        self.depth = depth
        self.slices = slices
        
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])
        
        
        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i), 
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate, 
                                                                       depth = dilated_layers)
        self.main_modules.append(self.bottleneck)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)),
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)), 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth, -1, -1)])
        
        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        self.main_modules.append(self.last)
        
        self.logvar = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        # down
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out).clone()
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)
        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        
        out = self.bottleneck(out)
        
        links.reverse()

        # up
        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        logvar = self.logvar(out)
        logvar = -torch.exp(logvar)

        return torch.unsqueeze(pred, 1) , torch.unsqueeze(logvar, 1)
    