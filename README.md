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

## 示例索引

| 方向 | 示例文件 | 主要内容 | 数据要求 |
| --- | --- | --- | --- |
| KNN | `KNN_分类思路.py`、`KNN_回归思路.py` | KNN 分类与回归的基本原理 | 无外部数据 |
| KNN 案例 | `KNN_鸢尾花案例.py` | 数据划分、标准化、训练与准确率评估 | scikit-learn 内置鸢尾花数据集 |
| 特征处理 | `特征值预处理_归一化.py`、`特征预处理_标准化.py` | Min-Max 归一化与 Z-score 标准化 | 无外部数据 |
| 模型选择 | `网格搜索和交叉验证.py` | KNN 超参数搜索与交叉验证 | scikit-learn 内置鸢尾花数据集 |
| 回归 | `线性回归API入门.py`、`波士顿房价预测-正规方程法.py` | 线性回归、误差评估与正则化 | 房价案例运行时需要网络 |
| 分类 | `逻辑回归_癌症预测.py`、`逻辑回归_电信流失用户预测.py` | 二分类流程和分类指标 | 需要对应 CSV 文件 |
| 树模型 | `CART分类_泰坦尼克号案例.py`、`随机森林算法_代码演示.py` | 决策树、随机森林和参数调优 | 需要泰坦尼克号 CSV 文件 |
| 集成学习 | `AdaBoost算法_葡萄酒案例.py` | AdaBoost 分类流程 | 需要葡萄酒 CSV 文件 |
| PyTorch 基础 | `TestPyTorch.py`、`nn_module.py` | 环境检查和 `nn.Module` 基础 | 无外部数据 |
| 图像与网络层 | `transforms_test.py`、`nn.conv.py`、`nn.linear.py` | 图像变换、卷积层、线性层和 TensorBoard | 使用仓库图片或下载 CIFAR-10 |

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

机器学习示例的常用依赖：

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

PyTorch 示例还需要：

```text
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
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

如果要运行 PyTorch 示例，再安装：

```powershell
pip install pillow torch torchvision tensorboard
```

> PyTorch 的 CPU、CUDA 安装命令可能不同。若需要使用显卡，建议根据自己的设备参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 生成安装命令。

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

推荐的数据目录形式如下：

```text
MLStudy/
└── data/
    ├── breast-cancer-wisconsin.csv
    ├── churn.csv
    ├── titanic_train.csv
    ├── wine.csv
    ├── train.csv
    └── demo.png
```

外部数据文件的列名和格式需要与脚本中的读取、特征选择逻辑保持一致。体积较大的数据集、模型和运行日志通常不建议提交到 Git 仓库。

## 使用提示

- 文件名包含中文，建议使用 UTF-8 编码和支持中文路径的终端或 IDE。
- 部分脚本执行后会显示 Matplotlib 图形或生成 TensorBoard 日志。
- 示例代码重在展示学习思路，运行前可以打开脚本查看末尾实际调用了哪个函数。
- 数据集路径和输出路径均以脚本当前实现为准；建议从 `MLStudy` 或 `PytorchStudy` 子目录运行对应脚本。如果更改运行目录，需要同步调整相对路径。
- `nn.conv.py` 和 `nn.linear.py` 首次运行时会下载 CIFAR-10，请确保网络连接正常并预留数据存储空间。
- 如果绘图时中文显示异常，可在 Matplotlib 中配置本机已有的中文字体。

## 常见问题

### 提示找不到文件

先确认当前终端位于正确的子目录，并检查 `MLStudy/data` 中是否存在脚本需要的数据文件。可在 PowerShell 中运行 `Get-Location` 查看当前位置。

### TensorBoard 没有显示内容

先运行会写入日志的 PyTorch 脚本，再从 `PytorchStudy` 目录执行：

```powershell
tensorboard --logdir ../logs
```

### PyTorch 无法使用 CUDA

运行以下脚本检查环境：

```powershell
python TestPyTorch.py
```

输出为 `False` 时，程序仍可使用 CPU；如需 GPU，应检查 NVIDIA 驱动、CUDA 兼容性以及 PyTorch 安装版本。

## 学习目标

这个仓库用于逐步熟悉以下流程：

1. 读取并观察数据；
2. 清洗数据和处理特征；
3. 划分训练集与测试集；
4. 选择、训练并调节模型；
5. 使用合适的指标评估模型；
6. 理解 PyTorch 中数据、网络层和前向传播的基本工作方式。

欢迎通过 Issue 提出建议，或在自己的分支上继续补充学习案例。
