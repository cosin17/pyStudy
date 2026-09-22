from torch import nn
import torch

class KeWei(nn.Module):
    def __init__(self): # 初始化
        super(KeWei, self).__init__()

    def forward(self, input): # 前向传播
        output = input + 1
        return output

kewei = KeWei()
x = torch.tensor(1.0)
output = kewei(x)
print(output)
