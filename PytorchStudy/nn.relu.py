from torch import nn
import torch
from torch.nn import ReLU

input = torch.tensor([[1,-0.5],
                      [-1,3]])
input = torch.reshape(input, (-1,1,2,2))
print(input.shape)

class KeWei(nn.Module):
    def __init__(self):
        # 初始化父类的参数
        # super(KeWei, self) 找到keWei的父类，然后用self这个实例去调用父类的方法
        super(KeWei, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output

kewei = KeWei()
output = kewei(input)
print(output)

