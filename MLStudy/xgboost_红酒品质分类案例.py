"""
案例：
    通过XGBoost极限梯度提升树 完成 红酒品质分类案例

回顾：XGBoost 极限梯度提升树
    概述：
        Extreme Gradient Boosting Tree，底层采用 打分函数 决定是否分枝
    原理：
        Gain值 = 分枝前的打分-（分枝后左子树打分+分枝后右子树打分）
        如果Gain值>0，考虑分支，否则不考虑分支

"""

import joblib                                           # 保存和加载模型
import numpy as np
import pandas as pd
import xgboost as xgb                                   # 极限梯度提升树对象
from collections import Counter                         # 统计分类结果
from sklearn.model_selection import train_test_split, GridSearchCV  # 划分训练集和测试集
from sklearn.metrics import classification_report       # 分类报告对象
from sklearn.model_selection import StratifiedKFold    # 交叉验证对象，类似于 网格搜索时 cv=折数
from sklearn.utils import class_weight

from MLStudy.随机森林算法_代码演示 import gs_estimator


# 1.定义函数，对源数据拆分成训练集和测试集，并存储到csv文件
def dm01_data_split():
    # 1.加载数据集
    df = pd.read_csv('./data/winequality-red.csv')
    # df.info()
    # 2.抽取特征数据 和 标签数据
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1] - 3  # 最好一列是标签，默认范围是：[3,8] ——> [0,5]
    # 3.查看数据
    # print(x[:5])
    # print(y[:5])
    # print(f'查看 标签结果的分布情况是否均衡：{Counter(y)}')

    # 4.划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23,stratify=y)
    # 5.把上述的 训练集特征 和 标签数据拼接在一起，测试集特征 和 标签数据拼接在一起，最后写到文件中
    pd.concat([x_train, y_train], axis=1).to_csv('./data/winequality-red_train.csv', index=False) # 横向拼接 忽略索引
    pd.concat([x_test, y_test], axis=1).to_csv('./data/winequality-red_test.csv', index=False) # 横向拼接 忽略索引

# 2.定义函数 训练模型 保存模型
def dm02_train_model():
    # 1.加载训练集和测试集
    train_data = pd.read_csv('./data/winequality-red_train.csv')
    test_data = pd.read_csv('./data/winequality-red_test.csv')
    # 2.抽取特征数据 和 标签数据
    x_train = train_data.iloc[:, :-1] # 除了最后一列，其他列都是特征数据
    y_train = train_data.iloc[:, -1] # 最后一列是标签数据

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # 3.创建模型对象
    estimator = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        random_state=23,
        objective='multi:softmax' # 多分类问题,使用多分类模型
    )
    # 加入 平衡权重 因为数据集是不平衡的，所以需要加入 平衡权重 来平衡数据集
    class_weight.compute_sample_weight('balanced',y_train)
    # 4.训练模型
    estimator.fit(x_train, y_train)
    # 5.模型评估
    print(f'准确率：{estimator.score(x_test, y_test)}')
    # 6.模型保存
    joblib.dump(estimator, './model/xgboost_model.pkl')
    print('模型保存成功!')

# 3.定义函数 测试模型
def dm03_train_model():
    # 1.加载训练集和测试集
    train_data = pd.read_csv('./data/winequality-red_train.csv')
    test_data = pd.read_csv('./data/winequality-red_test.csv')
    # 2.抽取特征数据 和 标签数据
    x_train = train_data.iloc[:, :-1] # 除了最后一列，其他列都是特征数据
    y_train = train_data.iloc[:, -1] # 最后一列是标签数据

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # 3.加载模型
    estimator = joblib.load('./model/xgboost_model.pkl')
    # 4.创建网格搜索 + 交叉验证(结合分层采样数据),找模型最优参数组合
    # 4.1定义变量，记录 最优参数组合
    param_dict = {'max_depth':[2,3,5,6,7],'n_estimators':[30,50,100,150],'learning_rate':[0.2,0.3,1,1.3]}
    # 4.2 创建 分层采样对象
    # 参1：折数 参2：是否打乱数据 参3：随机种子值
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
    # 4.3 创建 网格搜索 + 交叉验证(结合分层采样数据)对象
    # 参1：模型对象 参2：参数字典 参3：分层采样对象
    gs_estimator = GridSearchCV(estimator=estimator, param_grid=param_dict,cv=skf)
    # 5.模型训练
    gs_estimator.fit(x_train, y_train)
    # 6.模型预测
    y_pred = gs_estimator.predict(x_test)
    print(f'预测值为：{y_pred}')
    # 7.打印模型评估系数
    print(f'最优估计器对象组合：{gs_estimator.best_estimator_}')
    print(f'最优评分：{gs_estimator.best_score_}')

# 4.测试
if __name__ == '__main__':
    # dm01_data_split()
    # dm02_train_model()
    dm03_train_model()
