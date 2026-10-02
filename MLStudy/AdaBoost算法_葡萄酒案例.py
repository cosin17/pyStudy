"""
案例：
    演示AdaBoost算法之 葡萄酒案例

AdaBoost算法介绍：
    它属于 Boosting思想 即：串行执行，每次使用全部样本 最后加权投票
    原理：
        1.使用全部样本，通过决策树模型(第一个弱分类器)进行训练，获取结果
            思路：
                预测正确 ——> 权重下降
                预测错误 ——> 权重上升
        2.使用更新后的权重，通过决策树模型(第二个弱分类器)进行训练，获取结果
        ...
"""

# 导包
import pandas as pd
from sklearn.preprocessing import LabelEncoder         # 标签编码器
from sklearn.model_selection import train_test_split   # 划分训练集和测试集
from sklearn.tree import DecisionTreeClassifier        # 决策树分类器
from sklearn.ensemble import AdaBoostClassifier        # AdaBoost分类器
from sklearn.metrics import accuracy_score             # 准确率

# 1.加载数据
df_wine = pd.read_csv('./data/wine.csv')
# df_wine.info()
print(df_wine['Class label'].unique()) # [1 2 3] 葡萄酒类别有3种 但是决策树只能识别 二叉树

# 2.数据预处理
# 2.1 从 标签列(Class label) 中，过滤掉1类别，剩下2，3类别
df_wine = df_wine[df_wine['Class label'] != 1]
# 2.2 获取特征列 和 标签列
x = df_wine[['Alcohol','Hue']]
y = df_wine['Class label']
# 2.3 对标签列 进行编码 [2,3] -> [0,1]
le = LabelEncoder()
y = le.fit_transform(y)
# 2.4 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y) # stratify=y 表示按类别比例划分

# 3.特征工程 此处略

# 4.模型训练、预测、评估

# 场景1：单一决策树 ——> 充当弱分类器
# 4.1 创建模型对象
estimator1 = DecisionTreeClassifier(max_depth=3)
# 4.2 模型训练
estimator1.fit(x_train, y_train)
# 4.3 模型预测
y_pred1 = estimator1.predict(x_test)
print(f'单一决策树预测值为：{y_pred1}')
# 4.4 模型评估
print(f'单一决策树准确率：{accuracy_score(y_test, y_pred1)}')

# 场景2：AdaBoost ——> 集成学习，CART树，200棵
# 4.1 创建模型对象
# 参1：弱分类器(决策树对象)  参2：弱分类器数量  参3：学习率
estimator2 = AdaBoostClassifier(estimator1, n_estimators=200, learning_rate=0.1)
# 4.2 模型训练
estimator2.fit(x_train, y_train)
# 4.3 模型预测
y_pred2 = estimator2.predict(x_test)
print(f'AdaBoost预测值为：{y_pred2}')
# 4.4 模型评估
print(f'AdaBoost准确率：{accuracy_score(y_test, y_pred2)}')

