import torchvision
from torch import nn
from torchvision import transforms

# train_data = torchvision.datasets.ImageNet("data_image_net",split='train',download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=None)                    # 不加载预训练权重
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')                # 加载默认预训练权重

print(vgg16_true)
# 参数2 ：是否加载预训练权重
train_data = torchvision.datasets.CIFAR10('dataset', train=True, transform=transforms.ToTensor(), download=True)
# 在classifier层中添加全连接层 参1：输入维度 参2：输出维度
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 在classifier层中替换全连接层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
