import torch
import torchvision
from torch import nn
from model_save import *

# 方式1 ——> 加载模型
# PyTorch 2.6 把 torch.load 的 weights_only 参数默认值从 False 改成了 True
model = torch.load("model/vgg16_method1.pth", weights_only=False)
# print(model)

# 方式2 ——> 加载模型参数
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("model/vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱
# class Kewei(nn.Module):
#     def __init__(self):
#         super(Kewei,self).__init__()
#         self.conv1 = nn.Conv2d(3,64,kernel_size=3)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         return x
model = torch.load("model/kewei_method1.pth", weights_only=False)
print(model)