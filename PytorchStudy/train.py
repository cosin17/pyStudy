import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# 准备数据集
train_data = torchvision.datasets.CIFAR10('dataset', train=True,
                                          transform=transforms.ToTensor(), download=True) # 训练集
test_data = torchvision.datasets.CIFAR10('dataset', train=False,
                                         transform=transforms.ToTensor(), download=True) # 测试集

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果 train_data_size=10,训练数据集的长度为10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Kewei(nn.Module):
    def __init__(self):
        super(Kewei, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(), # 64*4*4
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),
        )

    def forward(self,x):
        x = self.model(x)
        return x

# 创建网络模型
kewei = Kewei()

# 损失函数
loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数

# 优化器 随机梯度下降
optimizer = torch.optim.SGD(kewei.parameters(), lr=0.01)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = kewei(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad() # 清空梯度
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试步骤开始
        total_test_loss = 0
        total_accuracy = 0 # 测试准确率
        with torch.no_grad(): # 测试时，关闭梯度计算
            for data in test_dataloader:
                imgs, targets = data
                outputs = kewei(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item() # 测试损失
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy.item()
        print("整体测试集上的loss:{}".format(total_test_loss))
        print("整体测试集上的准确率:{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step += 1

        torch.save(kewei,"model/kewei_{}.pth".format(i))
        # torch.save(kewei.state_dict(), "model/kewei_{}.pth".format(i))
        print("模型已保存")
writer.close()
