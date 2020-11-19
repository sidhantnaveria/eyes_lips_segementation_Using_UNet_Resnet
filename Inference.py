# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:39:51 2020

@author: sidhant
"""
from model import ModelClass
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim
from torch.autograd import Variable

def predict(filepath):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model=ModelClass().to(device)
    #/content/drive/MyDrive/last_model_full.pkl
    #"/content/drive/MyDrive/Unet_resnet3.pkl"
    model.load_state_dict(torch.load("UNetWithResnet34.pkl",map_location=torch.device('cpu')))
    model.eval()
    image = cv2.resize(cv2.imread(filepath),(256,256))
    image_array=np.array(image)
    inp = Variable(torch.from_numpy(image_array).type(torch.float32))
    inp=inp.unsqueeze(0)
    output = model(inp.permute(0,3,1,2))
    out=output.permute(0,2,3,1).squeeze(0)
    out=out.squeeze(2)
    
    
    out=out.to('cpu').detach().numpy()
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(image_array)
    axarr[1].imshow(out)
    
    
predict("C:/Users/sidha/Downloads/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/28483.jpg")
    
