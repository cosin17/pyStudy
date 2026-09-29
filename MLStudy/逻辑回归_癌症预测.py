"""
案例：
    演示逻辑回归模型实现 癌症预测

逻辑回归模型介绍（Logistic Regression）：
    概述：
        属于有监督学习，即：有特征 有标签 且标签是离散的
        主要适用于：二分类
    原理：
        把线性回归处理后的预测值 ——> 通过 Sigmoid激活函数，映射到[0,1]区间 ——> 基于自定义的阈值，结合概率来分类
    损失函数：
        极大似然估计函数的 负数形式

回顾：机器学习项目流程
     1.加载数据
     2.数据的预处理
     3.特征工程（特征提取 特征预处理）
     4.模型训练
     5.模型预测
     6.模型评估
    """

# 导包
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.preprocessing import StandardScaler     # 标准化
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn.metrics import accuracy_score           # 模型评估

from MLStudy.网格搜索和交叉验证 import transfer

# 1.加载数据
data =  pd.read_csv('./data/breast-cancer-wisconsin.csv')
data.info() # 查看数据信息

# 2.数据的预处理
# 参1：要被替换的值 参2：替换值 参3：是否在原数据上操作
data.replace('?',np.nan,inplace=True)
# 缺失值处理 ——> 删除缺失值
data.dropna(axis=0,inplace=True) # axis=0 表示按行删除缺失值
# 打印处理后的信息
data.info()

# 3.特征工程（特征提取 特征预处理...）
# 3.1特征提取
x = data.iloc[:,1:-1] # : 表示所有行，1:-1 表示从第2列开始，到倒数第列结束
y = data.iloc[:,-1] # 获取最后一列
# y = data.iloc['Class'] # 同上
# y = data.Class # 同上
# 3.2查看特征 和 标签
print(x[:5])
print(y[:5])
print(x.shape,y.shape)
# 3.3切割训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)
# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# 3.4标准化
# 3.4.1 创建标准化对象
transfer = StandardScaler()
# 3.4.2 对训练集、测试集进行标准化，训练 + 标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.模型训练
# 4.1 创建模型对象(逻辑回归模型)
estimator = LogisticRegression()
# 4.2 训练模型
estimator.fit(x_train,y_train)

# 5.模型预测
y_pred = estimator.predict(x_test)
print(f'预测值为：{y_pred}')

# 6.模型评估
# 正确率，公式为：正确预测数 / 总样本数
print(f'预测前评估的正确率：{estimator.score(x_test,y_test)}')
print(f'预测后评估的正确率：{accuracy_score(y_test,y_pred)}')

# 思考：逻辑回归模型能用 准确率来评测吗？
# 答案：可以，但是结果不精准，因为逻辑回归模型主要用于 二分类，即：A类还是B类，不能说 97%的A类，3%的B类
# 所以要通过 混淆矩阵来测评，即：精确率，召回率，F1值(F1-Score)，ROC曲线,AUC值