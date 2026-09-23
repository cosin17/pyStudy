import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from nn_module import output

dataset = torchvision.datasets.CIFAR10("../MLStudy/data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64) # 批量大小

class Kewei(nn.Module):
    def __init__(self):
        super(Kewei, self).__init__()
        self.linear1 = nn.Linear(196608,10) # 输入层196608，输出层10

    def forward(self, input):
        output = self.linear1(input)
        return output

kewei = Kewei()

for data in dataloader:
    imgs,targets = data
    print(imgs,targets)
    output = torch.flatten(imgs) # 展平图片
    print(output.shape)
    output = kewei(output)
    print(output.shape)
