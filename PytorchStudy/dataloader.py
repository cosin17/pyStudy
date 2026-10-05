from tkinter.constants import SUNKEN

import  torchvision

# 准备的测试数据集
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=False)
# drop_last=True 表示最后一个批次不足时，不添加到数据加载器 num_workers=0 表示不使用多线程加载数据
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0,drop_last=True)

# 测试数据集中第一张图片及target
img,target = test_data[0]
print(img.shape)  # 输出：torch.Size([3, 32, 32])
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets.shape)
        writer.add_images("Epoch:{}".format(epoch),imgs,step) # add_images 记录多张图片
        step += 1

writer.close()