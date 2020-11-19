# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:33:48 2020

@author: sidhant
"""

from torchvision.models import resnet34 as res34
from torchsummary import summary
import torch
import torch.nn as nn
from torch.nn import init

def convrelu(in_channels, out_channels, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
	    
        nn.Conv2d(out_channels, out_channels, kernel),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()

        self.model= res34(pretrained=True,progress=False)
        # for child in self.model.children():
        #     for param in child.parameters():
        #         param.requires_grad = False
        

        self.base_layers = list(self.model.children()) 
        self.model_layers1set=nn.Sequential(*self.base_layers[:2])
        self.skip1=self.base_layers[2]
        self.model_layer2set=nn.Sequential(*self.base_layers[3:4])
        self.skip2=self.base_layers[4]
        self.skip3=self.base_layers[5]
        self.skip4=self.base_layers[6]
        self.output=self.base_layers[7]
        
        self.model.fc=Identity()
        self.model.avgpool=Identity()
        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv=nn.Conv2d(16,1,3,stride=1)
        self.sigm=nn.Sigmoid()
        self.convtranspose1=nn.ConvTranspose2d(512,256,2,stride=2)
        self.convtranspose2=nn.ConvTranspose2d(256,128,(3,3),stride=(3,3),padding=(2,2))
        self.convtranspose3=nn.ConvTranspose2d(128,64,(12,12),stride=(2,2),padding=(1,1))
        self.convtranspose4=nn.ConvTranspose2d(64,64,(10,10),stride=(2,2),padding=(0,0))
        self.convtranspose5=nn.ConvTranspose2d(32,32,(16,16),stride=(2,2))
        self.contrans1=convrelu(512,256,3)
        self.contrans2=convrelu(256,128,3)
        self.contrans3=convrelu(128,64,3)
        self.contrans4=convrelu(128,32,3)
        self.contrans5 =convrelu(32,16,3)
        
        
        
        
    def forward(self,x):
        out=self.model_layers1set(x)
        skip1=self.skip1(out)
        # print("skip1",skip1.shape)
        out=self.model_layer2set(skip1)
        skip2=self.skip2(out)
        # print("skip2",skip2.shape)
        skip3=self.skip3(skip2)
        # print("skip3",skip3.shape)
        skip4=self.skip4(skip3)
        # print("skip4 shape",skip4.shape)
        encoderOut=self.output(skip4)
        # print(encoderOut.shape)
        
        out=self.convtranspose1(encoderOut)
        # print("out",out.shape)
        out=torch.cat([out,skip4],dim=1)
        
        out=self.contrans1(out)
        
        out=self.convtranspose2(out)
        # print("cont2",out.shape)
        out=torch.cat([out,skip3],dim=1)
        out=self.contrans2(out)
        
        # print("out3",out.shape)
        out=self.convtranspose3(out)
        # print("cont3",out.shape)
        out=torch.cat([out,skip2],dim=1)
        out=self.contrans3(out)
        
        out=self.convtranspose4(out)
        # print("cont4",out.shape)
        out=torch.cat([out,skip1],dim=1)
        out=self.contrans4(out)
        
        out=self.convtranspose5(out)
        out=self.contrans5(out)
        out=self.conv(out)
        # out=self.sigm(out)
        # print(out.shape)
        return out
        
        




