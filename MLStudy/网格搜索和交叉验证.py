"""
案例：演示网格搜索 和 交叉验证

交叉验证解释：
   原理：
       把数据分成n份 ——> n折交叉验证
       第1次:把第1份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取准确率
       第2次:把第2份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取准确率
       第3次:把第3份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取准确率
       ...
       然后计算上述准确率的平均值，作为模型的准确率

       假设第4次最好，则用全部数据（训练集+测试集）训练模型，作为最终模型
   目的：
       为了让模型的准确率更准确

网格搜索：
   目的：
       为了找到最佳的超参数组合，使模型的准确率最高
   原理：
       接收超参可能出现的值，然后针对于 超参的每个值进行 交叉验证,找到最优超参组合
   超参数：
       需要用户手动录入数据，不同的超参组合，会导致不同的模型准确率
大白话解释：
    网格搜索 + 交叉验证，本质上指的是 GridSearchCV 这个API，它会帮我们寻找最优超参
"""

# 导入鸢尾花数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV # 导入训练集测试集划分函数 寻找最优超参的网格搜索类
from sklearn.preprocessing import StandardScaler                   # 导入标准化类
from sklearn.neighbors import KNeighborsClassifier                 # 导入KNN分类器
from sklearn.metrics import accuracy_score                         # 导入准确率评估指标

# 1. 加载鸢尾花数据集
iris_data = load_iris()

# 2. 数据预处理，这里是：切分训练集和测试集，比例 8：2
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)

# 3.特征工程 ——> 特征预处理 ——> 标准化
#创建标准化对象
transfer = StandardScaler()
#对训练集和测试集的特征数据进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.模型训练
#创建KNN分类器对象
estimator = KNeighborsClassifier()
#定义字典，记录超参可能出现的情况
param_dict = {"n_neighbors":[i for i in range(1,11)]} # i 1-10
#创建 GridSearchCV 对象 ——> 使用网格搜索 + 交叉验证
# 参1：要计算最优超参的模型对象
# 参2：该模型超参可能出现的值
# 参3：交叉验证折数 4*10=40次
# 返回值 estimator 处理后的模型对象
estimator = GridSearchCV(estimator=estimator, param_grid=param_dict,cv=4)
estimator.fit(x_train, y_train)

print(f'最优评分：{estimator.best_score_}')
print(f'最优超参：{estimator.best_params_}')
print(f'最优估计器对象：{estimator.best_score_}')
print(f'具体的交叉验证结果：{estimator.cv_results_}')

# 5.模型评估
# 获取最优超参的 模型对象
best_estimator = estimator.best_estimator_
# 模型训练
best_estimator.fit(x_train, y_train)
# 模型预测
y_pred = best_estimator.predict(x_test)

print(f'准确率：{accuracy_score(y_test, y_pred)}')
