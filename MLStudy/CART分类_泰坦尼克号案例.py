"""
案例：演示 CART 分类回归决策树的 分类功能
"""

# 导包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# 1.加载数据
data = pd.read_csv('./data/titanic_train.csv')
# data.info()
# print(data.head(5))

# 2.数据预处理
# 2.1 提取特征和标签
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']
# print(x.head(5),y.head(5))
# 2.2 发现Age有缺失值，我们用该列 平均值做填充
# x['Age'] = x['Age'].fillna(x['Age'].mean(), inplace=True) # 会报警告 但是可以用
# x['Age'] = x['Age'].fillna(x['Age'].mean())               # 会报警告 因为直接修改原数据了
# 解决方案，copy()数据后再改
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3 针对于Sex列 进行one-hot编码
x = pd.get_dummies(x, columns=['Sex'])
# x.info()
# 2.4 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3.特征工程

# 4.模型训练
estimator = DecisionTreeClassifier(max_depth=10) # max_depth=10 表示最大深度为10，防止过拟合
estimator.fit(x_train, y_train)

# 5.模型预测
y_pred = estimator.predict(x_test)
print(f'预测值为：{y_pred}')

# 6.模型评估
print(f'分类评估报告:\n{classification_report(y_test, y_pred)}')
# 7.绘制 决策树
plt.figure(figsize=(200, 100)) # 设置图片大小 30*100(dpi) * 20*100(dpi) = 3000 * 2000像素
plot_tree(estimator,filled=True,max_depth=10) # filled=True 表示填充颜色
plt.savefig('./data/titanic_tree.png')
plt.show()