from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

# 加载数据集
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=ToTensor(),
                                       download=True )
dataloader = DataLoader(dataset, batch_size=64) # 批量大小
class Kewei(nn.Module):
    def __init__(self):
        super(Kewei, self).__init__()
        # 卷积层1
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

kewei = Kewei()
print(kewei)

writer = SummaryWriter("../logs")
step = 0

for data in dataloader:
    imgs,targets = data # 解包：图片(batch_size,3,32,32) + 标签
    output = kewei(imgs) # 前向传播（自动触发 forward 方法）
    print(output.shape)
    writer.add_image("input",imgs,step)

    output = torch.reshape(output,(-1,3,30,30))
    writer.add_image("output",output,step)
    step += 1

writer.close()
