"""
案例：
    演示 集成学习之 Bagging思想 随机森林算法 代码

集成学习：
    概述：
        把多个弱学习器 组成1个强学习器，通过投票或平均的方式，实现分类或回归的功能
    思想：
        Bagging思想：
            1.有放回的随机抽样
            2.平均投票
            3.可以并行执行

        Boosting思想：
            1.每次训练都会使用全部样本
            2.加权投票 ——> 预测正确：权重降低 预测错误：权重增加
            3.只能串行执行
        Bagging思想代表：
            随机森林算法

随机森林算法：
    1.每个弱学习器都是 CART决策树（必须是二叉树）
    2.有放回的随机抽样 平权投票 并行执行
"""

# 导包
import pandas as pd
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.tree import DecisionTreeClassifier       # 决策树分类器
from sklearn.ensemble import RandomForestClassifier   # 随机森林分类器
from sklearn.model_selection import GridSearchCV      # 网格搜索

# 1.加载数据
df = pd.read_csv('./data/titanic_train.csv')

# 2.数据预处理
# 2.1 提取特征和标签
x = df[['Pclass', 'Sex', 'Age']].copy()
y = df['Survived']
# 2.2 空值处理
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 2.3 热编码处理
x  = pd.get_dummies(x)
# 2.4 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
# 3.特征工程

# 4.模型训练、预测、评估

# 场景1：单一决策树
# 4.1 创建 决策树对象，演示：单一的决策树效果
estimator1 = DecisionTreeClassifier()
# 4.2 模型训练
estimator1.fit(x_train, y_train)
# 4.3模型预测
y_pred = estimator1.predict(x_test)
print(f'预测值为：{y_pred}')
# 4.4模型评估
print(f'决策树模型的准确率为：{estimator1.score(x_test, y_test)}') # score()方法返回模型在测试集上的准确率 0.8044692737430168
print('-'*23)

# 场景2： 随机森林算法 ——> 采用默认参数
# 4.1 创建 随机森林对象，演示：多个的决策树(Bagging)效果
estimator2 = RandomForestClassifier()  # n_estimators=100,max_depyh=None
# 4.2 模型训练
estimator2.fit(x_train, y_train)
# 4.3 模型预测
y_pred2 = estimator2.predict(x_test)
print(f'预测值为：{y_pred2}')
# 4.4模型评估
print(f'随机森林模型的准确率为：{estimator2.score(x_test, y_test)}')

# 场景3：随机森林算法 ——> 采用网格搜索
# 4.1 创建 随机森林对象，演示：多个的决策树(Bagging)效
estimator3 = RandomForestClassifier()
# 4.2 参数准备
params = {'n_estimators':[30,50,60,90,110],'max_depth':[2,3,5,7]}
# 4.3 创建网格搜索对象 结合 交叉验证
gs_estimator = GridSearchCV(estimator3, params, cv=3)
# 4.4 模型训练
gs_estimator.fit(x_train, y_train)
# 4.5 模型预测
y_pred3 = gs_estimator.predict(x_test)
print(f'预测值为：{y_pred3}')
# 4.6模型评估
print(f'随机森林模型的准确率为：{gs_estimator.score(x_test, y_test)}')
# 4.7 最佳参数
print(f'最佳参数为：{gs_estimator.best_params_}')
