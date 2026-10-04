"""
案例：
    演示KMeans 聚类算法 入门案例

KMeans简介：
    它属于 无监督学习，即：有特征，无标签，根据样本间的相似性进行划分
    所谓的相似性 可以理解为 就是 距离，例如：欧氏距离，曼哈顿距离，余弦相似度等

    一般大厂，项目初期在没有 先备知识(标签) 的情况下，可能会用
"""
import os
os.environ['OMP_NUM_THREADS'] = '3'  # OpenMP多任务程序 3个线程并行计算
# 导包
from sklearn.cluster import KMeans                   # 聚类的API，采用指定 质心 来分簇
import matplotlib.pyplot as plt                      # 绘图的
from sklearn.datasets import make_blobs              # 默认会按照高斯分布(正态分布)生成数据集，只需要指定 均值 标准差
from sklearn.metrics import calinski_harabasz_score  # 评价指标，值越大，聚类效果越好


# 1. 生成数据集
# 参1：样本数量  参2：特征数量  参3：标签数量  参4：簇内标准差  参5：随机种子
x,y = make_blobs(n_samples=1000, n_features=2,centers=[[-1,-1],[1,1],[0,0],[2,2]], cluster_std=[0.4,0.2,0.3,0.4], random_state=23)
# print(x)
# print(y)

# 2.绘制上述的模型
# 参1：x轴数据  参2：y轴数据
plt.scatter(x[:,0],x[:,1])
plt.show()

# 3.创建KMeans对象
estimator = KMeans(n_clusters=4,random_state=23)

# 4.模型训练 和预测
y_pred = estimator.fit_predict(x)

# 5.绘制预测结果
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()

# 6.评价指标
print(f'评价指标：{calinski_harabasz_score(x,y_pred)}') # 越大越好
