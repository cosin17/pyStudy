# pyStudy

这是一个用于记录 Python 机器学习与 PyTorch 学习过程的示例仓库。项目以可直接阅读和运行的脚本为主，包含常见机器学习算法、数据预处理、模型评估，以及 PyTorch 张量、图像变换和神经网络模块等基础内容。

> 本仓库主要用于个人学习和实验。部分脚本依赖本地数据文件，代码也会随着学习进度持续调整，不建议直接作为生产项目使用。

## 学习内容

### 机器学习

`MLStudy` 目录目前包括：

- KNN 分类、回归与鸢尾花分类案例
- 特征归一化与标准化
- 线性回归与波士顿房价预测
- L1、L2 正则化与过拟合处理
- 逻辑回归、癌症预测与电信用户流失预测
- 混淆矩阵、精确率、召回率和 F1 值
- 网格搜索与交叉验证
- 决策树、随机森林和 AdaBoost
- 手写数字识别示例

### PyTorch

`PytorchStudy` 目录目前包括：

- PyTorch 与 CUDA 环境检查
- TensorBoard 基础使用
- PIL 图像、NumPy 数组与 Tensor 的转换
- `torchvision.transforms` 常见图像变换
- `nn.Module`、线性层和卷积层基础
- CIFAR-10 数据集的加载与简单前向传播

## 项目结构

```text
pyStudy/
├── MLStudy/           # scikit-learn 机器学习示例
├── PytorchStudy/      # PyTorch 学习示例
├── hymenoptera_data/  # 蚂蚁与蜜蜂图像数据
├── images/            # 图像变换示例使用的图片
├── .idea/             # PyCharm 配置
└── .vscode/           # VS Code 配置
```

## 环境要求

- Python 3.9 或更高版本
- 建议使用虚拟环境
- PyTorch 是否支持 CUDA 取决于显卡、驱动和安装版本；没有 GPU 时也可以使用 CPU 运行多数示例

常用依赖如下：

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
pillow
torch
torchvision
tensorboard
```

可以根据需要安装完整环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib seaborn scikit-learn joblib pillow torch torchvision tensorboard
```

> PyTorch 的 CPU、CUDA 安装命令可能不同，建议根据自己的设备参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

## 运行示例

克隆仓库：

```bash
git clone https://github.com/cosin17/pyStudy.git
cd pyStudy
```

部分脚本使用相对于子目录的文件路径。运行机器学习示例时，建议先进入 `MLStudy`：

```bash
cd MLStudy
python "KNN_鸢尾花案例.py"
python "网格搜索和交叉验证.py"
python "AdaBoost算法_葡萄酒案例.py"
```

运行 PyTorch 示例时，回到仓库根目录并进入 `PytorchStudy`：

```bash
cd ../PytorchStudy
python "TestPyTorch.py"
python "nn.conv.py"
```

查看 TensorBoard 日志：

```bash
tensorboard --logdir ../logs
```

然后按照终端提示，在浏览器中打开对应地址。

## 数据集说明

仓库中已经包含 `hymenoptera_data`，供部分图像与 transforms 示例使用。部分 `MLStudy` 脚本还会从 `MLStudy/data` 读取 CSV 或图片文件，例如：

- `breast-cancer-wisconsin.csv`
- `churn.csv`
- `titanic_train.csv`
- `wine.csv`
- `train.csv`
- `demo.png`

如果本地没有相应文件，这些脚本会出现文件不存在的错误，需要先准备数据并放入 `MLStudy/data`。CIFAR-10 示例设置了 `download=True`，首次运行时需要网络连接并会自动下载数据。

## 使用提示

- 文件名包含中文，建议使用 UTF-8 编码和支持中文路径的终端或 IDE。
- 部分脚本执行后会显示 Matplotlib 图形或生成 TensorBoard 日志。
- 示例代码重在展示学习思路，运行前可以打开脚本查看末尾实际调用了哪个函数。
- 数据集路径和输出路径均以脚本当前实现为准；建议从 `MLStudy` 或 `PytorchStudy` 子目录运行对应脚本。如果更改运行目录，需要同步调整相对路径。

## 学习目标

这个仓库用于逐步熟悉以下流程：

1. 读取并观察数据；
2. 清洗数据和处理特征；
3. 划分训练集与测试集；
4. 选择、训练并调节模型；
5. 使用合适的指标评估模型；
6. 理解 PyTorch 中数据、网络层和前向传播的基本工作方式。

欢迎通过 Issue 提出建议，或在自己的分支上继续补充学习案例。
