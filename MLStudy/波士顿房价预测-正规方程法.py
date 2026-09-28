"""
波士顿房价预测-正规方程法

回顾：
    线性回归算法 属于 有监督学习之 有特征，有标签，且标签是连续的
    分类：
       一元线性回归：1个特征列 + 1个标签列
       多元线性回归：多个个特征列 + 1个标签列
    公式：
       一元线性回归：
           y = kx + b => wx + b
               k：数学中叫斜率，在机器学习中Weigh(权重),简称：w
               b：数学中叫截距，在机器学习中Bias(偏置),简称：b
       多元线性回归：
           y = w1x1 + w2x2 + ... + wn*xn + b = w的转置 * x + b
机器学习开发流程：
    1.加载数据
    2.数据的预处理
    3.特征工程（特征提取，特征预处理...）
    4.模型选择
    5.模型训练
    6.模型评估
    7.模型部署
"""

# 导包
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 随机梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估
from sklearn.linear_model import Ridge,RidgeCV

import io
import urllib.request

import pandas as pd
import numpy as np

# 1.加载数据
# 说明：scikit-learn 1.2 起已移除 load_boston()，数据不再随包提供；
#       且 pandas 3.x 已不支持 read_csv 直接读 URL，因此改为 urllib 拉取后解析。
def _fetch(url):
    """从网站拉取文本内容（带 User-Agent，避免被服务器拦截）。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")

def load_boston():
    """从网站直接拉取波士顿房价数据，返回 (特征矩阵 data, 标签 target)。"""
    try:
        # 首选：CMU 统计系官方数据源（与 sklearn 原数据集一致，506 条样本）
        raw = pd.read_csv(io.StringIO(_fetch("http://lib.stat.cmu.edu/datasets/boston")),
                          sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
        target = raw.values[1::2, 2]
        return data, target
    except Exception:
        # 备用：GitHub 镜像（带表头的 CSV：13 个特征 + MEDV）
        df = pd.read_csv(io.StringIO(
            _fetch("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")))
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

data, target = load_boston()
print(f'{data[:5]}')
print(f'标签前5个: {target[:5]}')

# 2.数据的预处理
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

# 3.特征工程（特征提取，特征预处理...）
# 创建标准化对象
transfer = StandardScaler()
#对训练集进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.模型训练
# 创建 线性回归 正规方程 模型对象
estimator = LinearRegression(fit_intercept=True) # fit_intercept:是否需要截距，默认是True
# 训练模型
estimator.fit(x_train, y_train)
# 打印计算出的 w 和 b
print(f'权重：{estimator.coef_}')
print(f'偏置：{estimator.intercept_}')

# 5.模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果：{y_pred}')

# 6.模型评估
# 均方误差评估
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差：{mse}')
# 均方根误差评估
rmse = root_mean_squared_error(y_test, y_pred)
print(f'均方根误差：{rmse}')
# 平均绝对误差评估
mae = mean_absolute_error(y_test, y_pred)
print(f'平均绝对误差：{mae}')
