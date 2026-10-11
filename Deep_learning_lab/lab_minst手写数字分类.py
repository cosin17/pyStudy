"""
MNIST 手写数字分类 —— PyTorch 完整流程整合版
=================================================
功能：数据准备 → 模型定义 → 训练 → 评估 → 保存 → 推理演示
运行：python mnist_main.py
依赖：torch、torchvision（需先安装，见 README）
数据集：自动读取本文件所在目录下 datas 文件夹中已下载的 MNIST 数据
"""

import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms,datasets



# ==================== 1. 超参数配置 ====================
BATCH_SIZE = 64          # 每批送入网络的样本数
EPOCHS = 10              # 完整训练轮数
LR = 1e-3                # 学习率（Adam 优化器的步长）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集根目录：指向本脚本所在目录下的 datas 文件夹，
# 用 __file__ 拼绝对路径，保证从任何目录启动都能找到数据
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datas")

# ==================== 2. 数据准备与加载 ====================
# 预处理流水线：先转 Tensor，再做标准化
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
train_set = datasets.MNIST(root=DATA_ROOT, train=True,download=True,transform=transform)
test_set = datasets.MNIST(root=DATA_ROOT, train=False,download=True,transform=transform)

# 数据加载
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# ==================== 3. 模型定义 ====================
class Net(nn.Module):
    """简单 CNN：两个卷积块提取特征 + 两个全连接层分类。

    输入单通道 28x28 灰度图，输出 10 个类别的得分（logits）。
    特征图尺寸变化：28 -> 14 -> 7（每次 MaxPool2d 减半）。
    """
    def __init__(self):
        super(Net, self).__init__()
        # 卷积部分：提取局部特征
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 全连接部分：把特征展平后做分类
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self,x):
        """前向传播：x 形状为 [B, 1, 28, 28] -> 返回 [B, 10]"""
        x = self.fc(self.conv(x))
        return x

# ==================== 4. 训练与评估函数 ====================
def train_one_epoch(model,criterion,optimizer,loader):
    """训练一个epoch：对每个batch执行 前向 —> 算损失 —> 反向传播 —> 更新参数"""
    model.train() #切换到训练模式（影响Dropout、BN 等层行为）
    total_loss,correct,total = 0,0,0
    for images,labels in loader:
        images,labels = images.to(DEVICE),labels.to(DEVICE)

        optimizer.zero_grad()              # 1) 清空上一批累计的梯度
        outputs = model(images)            # 2) 前向传播，得到预测得分
        loss = criterion(outputs, labels)  # 3) 计算交叉熵损失
        loss.backward()                    # 4) 反向传播，求出各参数梯度
        optimizer.step()                   # 5) 按梯度更新参数

        # 统计：损失和正确数按样本数加权，便于求平均
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad() # 评估不需要梯度：省显存、提速
def evaluate(model,loader):
    """评估模型在测试集上的准确率"""
    model.eval() # 设置为评估模式，关闭dropout和BN
    correct = total = 0
    for images,labels in loader:
        images,labels = images.to(DEVICE),labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item() # 取得分最高的类别
        total += labels.size(0) # 样本总数
    return correct / total




# ==================== 5. 主流程 ====================
def main():
    # 初始化模型、损失函数、优化器
    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()                         # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # Adam 优化器（自适应学习率）

    print(f'设备: {DEVICE} | 训练集: {len(train_set)} 张 | 测试集: {len(test_set)} 张')

    # ---- 训练循环 ----
    for epoch in range(1, EPOCHS + 1):
        loss,acc = train_one_epoch(model,criterion,optimizer,train_loader)
        test_acc = evaluate(model,test_loader) # 每轮在测试集上验证一次
        print(f'Epoch {epoch:2d}/{EPOCHS} | 训练损失 {loss:.4f} | '
              f'训练准确率 {acc:.4f} | 测试准确率 {test_acc:.4f}')

    # ---- 保存模型权重 ----
    # 只保存state_dict（参数），体积小，加载时需重建相同结构
    os.makedirs("model", exist_ok=True)                     # 目录不存在就创建，存在则忽略
    torch.save(model.state_dict(),"model/mnist_cnn.pth")
    print("模型已保存 -> model/mnist_cnn.pth")

    # ---- 推理演示：取测试集前 5 张图预测并对比真实标签 ----
    model.eval()
    images,labels = next(iter(test_loader))
    images,labels = images[:5].to(DEVICE),labels[:5]
    with torch.no_grad():
        preds = model(images).argmax(1)
    for i in range(5):
        ok = "正确" if preds[i].item() == labels[i].item() else "错误"
        print(f'样本 {i+1}: 预测 {preds[i].item()} | 真实 {labels[i].item()} | 预测{ok}')



if __name__ == "__main__":
    main()