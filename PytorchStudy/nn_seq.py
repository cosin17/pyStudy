import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d,Flatten,Linear

class KeWei(nn.Module):
    def __init__(self):
        super(KeWei, self).__init__()
        # self.conv1 = Conv2d(3,32,5,padding=2) #卷积
        # self.maxpool1 = MaxPool2d(2) #池化
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten() #展平
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)
        # Sequential 顺序容器 用于将多个层按顺序组合起来
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64, 10)
        )
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        x= self.model1(x)
        return x

kewei = KeWei()
print(kewei)
input = torch.ones((64,3,32,32)) # torch.ones() 生成一个全1的张量
output = kewei(input)
print(output.shape)