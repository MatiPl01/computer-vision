#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
#print(conv)
#print('\n')



conv.state_dict()

conv.state_dict()['weight'][0][0]=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
conv.state_dict()['bias'][0]=0.0
conv.state_dict()


image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
print('image={}'.format(image))
print('\n')


z=conv(image)
print('z={}'.format(z))
print('\n')

# Determining the Size of the Output
# Let M be the size of the input and K be the size of the kernel. 
# The size of the output is given by the following formula:
# M_new = M - K + 1

K=2

conv1 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=K)

conv1.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv1.state_dict()['bias'][0]=0.0

conv1.state_dict()
print('conv1={}'.format(conv1))
print('\n')



M=4

image1=torch.ones(1,1,M,M)

z1=conv1(image1)

print("The activation map:",z1)
print("The shape of the activation map:",z1.shape[2:4])
print("The shape of the activation map:",z1.shape)
print('\n')
#exit()


conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

conv3.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv3.state_dict()['bias'][0]=0.0

conv3.state_dict()


z3=conv3(image1)

print("The activation map:",z3)
print("The shape of the activation map:",z3.shape[2:4])
print("The shape of the activation map:",z3.shape)


print('image1={}'.format( image1))
print('\n')



conv4 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3)

conv4.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv4.state_dict()['bias'][0]=0.0

conv4.state_dict()

z4 = conv4(image1)

print("z4:",z4)
print("z4:",z4.shape[2:4])
print("z4:",z4.shape)



conv5 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3,padding=1)

conv5.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv5.state_dict()['bias'][0]=0.0
conv5.state_dict()

z5 = conv5(image1)

print("z5:",z5)
print("z5:",z4.shape[2:4])
print("z5:",z4.shape)
