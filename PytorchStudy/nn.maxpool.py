import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)

# 为什么要reshape 因为nn.MaxPool2d要求输入张量必须是4维(batch_size,channel,height,width)
input = torch.reshape(input, (-1,1,5,5))
print(input.shape)

class Kewei(nn.Module):
    def __init__(self):
        super(Kewei, self).__init__()
        # kernel_size=3 表示池化核的大小为3*3
        # ceil_mode=True 表示池化核的大小为奇数时，是否向上取整
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

kewei = Kewei()
output = kewei(input)
print(output)