"""
案例：
    演示 欠拟合，正好拟合，过拟合，L1正则化，L2正则化的效果图

回顾：
    欠拟合：模型在训练集 和 测试集表现效果都不好（模型参数过少）
    正好拟合：模型在训练集 和 测试集表现效果都好
    过拟合：模型在训练集表现好，测试集表现不好（模型参数过多）

L1和L2正则化介绍：
    目的/思路：
            都是基于 惩罚系数 来修改（特征列）权重的，惩罚系数越大，对应的权重就越小
    区别：
        1.L1正则化：对特征列的权重进行绝对值惩罚，会将权重设为0
        2.L2正则化：对特征列的权重进行平方惩罚，不会将权重设为（无限趋近）0
"""

# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 随机梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估
from sklearn.linear_model import Ridge,Lasso            # L1、L2正则化模型

# 1.定义函数，模拟：欠拟合
def dm01_under_fitting():
    # 1.准备数据集（欠拟合）
    # 指定随机种子，则每次生成的数据都是固定的
    np.random.seed(23)
    # 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3,3,size=100) # 参1：最小值 参2：最大值 参3：样本数量
    # 基于x轴值，通过线性公式，生成y轴 100个数据，模拟：目标值 y = kx + b = 0.5*x**2 + x + 2 + 噪声
    y = 0.5*x**2 + x + 2 + np.random.normal(0,1,size=100)

    # 2.数据预处理，把x轴（特征）转成 多行一列的形式
    x = x.reshape(-1,1) # 转为单列
    # print(f'处理后的数据：{x}')

    # 3.特征工程，这里不做了，直接用100条数据，先训练后预测

    # 4.模型训练
    # 创建模型对象
    estimator = LinearRegression()  # 正规方程 线性回归模型
    # 模型训练
    estimator.fit(x,y)
    # 5.模型预测
    y_pred = estimator.predict(x)
    # 6.模型评估
    print(f'均方误差为：{mean_squared_error(y, y_pred)}')
    # 7.绘图
    plt.scatter(x,y,label='真实值')
    plt.plot(x,y_pred,label='预测值')
    plt.show()
# 2.定义函数，模型：正好拟合
def dm02_just_fitting():
    # 1.准备数据集（正好拟合）
    # 指定随机种子，则每次生成的数据都是固定的
    np.random.seed(23)
    # 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值 参2：最大值 参3：样本数量
    # 基于x轴值，通过线性公式，生成y轴 100个数据，模拟：目标值 y = kx + b = 0.5*x**2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2.数据预处理，把x轴（特征）转成 多行一列的形式
    x = x.reshape(-1, 1)  # 转为单列
    # 因为目前特征列只有1列，模型过于简单，会出现欠拟合问题，我们增加1列 从而增加模型复杂度
    x2 = np.hstack([x,x**2]) # 该函数的作用：将多个数组按列方向拼接
    # print(f'处理后的数据：{x}')

    # 3.特征工程，这里不做了，直接用100条数据，先训练后预测

    # 4.模型训练
    # 创建模型对象
    estimator = LinearRegression()  # 正规方程 线性回归模型
    # 模型训练
    estimator.fit(x2, y)
    # 5.模型预测
    y_pred = estimator.predict(x2)
    # 6.模型评估
    print(f'均方误差为：{mean_squared_error(y, y_pred)}')
    # 7.绘图
    # plt.scatter(x, y, label='真实值')
    # plt.plot(np.sort(x), y_pred[np.argsort(x)], label='预测值') # sort() 对x轴进行排序 默认升序 argsort() 返回排序后的索引
    x_flat = x.ravel()  # 把二维数组展平成一维 (100, 1) → (100,)
    sorted_indices = np.argsort(x_flat)  # 获取排序后的索引
    plt.scatter(x_flat, y, label='真实值')
    plt.plot(x_flat[sorted_indices], y_pred[sorted_indices], label='预测值')
    plt.legend()
    plt.show()
# 3.定义函数，模型：过拟合
def dm03_over_fitting():
# 1.准备数据集（正好拟合）
    # 指定随机种子，则每次生成的数据都是固定的
    np.random.seed(23)
    # 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值 参2：最大值 参3：样本数量
    # 基于x轴值，通过线性公式，生成y轴 100个数据，模拟：目标值 y = kx + b = 0.5*x**2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2.数据预处理，把x轴（特征）转成 多行一列的形式
    x = x.reshape(-1, 1)  # 转为单列
    # 因为目前特征列只有1列，模型过于简单，为了模拟过拟合，我们增加9列 从而增加模型复杂度
    x3 = np.hstack([x,x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9,x**10]) # 该函数的作用：将多个数组按列方向拼接
    # print(f'处理后的数据：{x}')

    # 3.特征工程，这里不做了，直接用100条数据，先训练后预测

    # 4.模型训练
    # 创建模型对象
    estimator = LinearRegression()  # 正规方程 线性回归模型
    # 模型训练
    estimator.fit(x3, y)
    # 5.模型预测
    y_pred = estimator.predict(x3)
    # 6.模型评估
    print(f'均方误差为：{mean_squared_error(y, y_pred)}')
    # 7.绘图
    # plt.scatter(x, y, label='真实值')
    # plt.plot(np.sort(x), y_pred[np.argsort(x)], label='预测值') # sort() 对x轴进行排序 默认升序 argsort() 返回排序后的索引
    x_flat = x.ravel()  # 把二维数组展平成一维 (100, 1) → (100,)
    sorted_indices = np.argsort(x_flat)  # 获取排序后的索引
    plt.scatter(x_flat, y, label='真实值')
    plt.plot(x_flat[sorted_indices], y_pred[sorted_indices], label='预测值')
    plt.legend()
    plt.show()
# 4.定义函数，模拟：L1正则化
def dm04_l1_regression():
    # 1.准备数据集（正好拟合）
    # 指定随机种子，则每次生成的数据都是固定的
    np.random.seed(23)
    # 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值 参2：最大值 参3：样本数量
    # 基于x轴值，通过线性公式，生成y轴 100个数据，模拟：目标值 y = kx + b = 0.5*x**2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2.数据预处理，把x轴（特征）转成 多行一列的形式
    x = x.reshape(-1, 1)  # 转为单列
    # 因为目前特征列只有1列，模型过于简单，为了模拟过拟合，我们增加9列 从而增加模型复杂度
    x3 = np.hstack([x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10])  # 该函数的作用：将多个数组按列方向拼接
    # print(f'处理后的数据：{x}')

    # 3.特征工程，这里不做了，直接用100条数据，先训练后预测

    # 4.模型训练
    # 创建L1正则化对象
    estimator = Lasso(alpha=0.1)  # alpha：惩罚系数 默认：0.1

    # 模型训练
    estimator.fit(x3, y)
    # 5.模型预测
    y_pred = estimator.predict(x3)
    # 6.模型评估
    print(f'均方误差为：{mean_squared_error(y, y_pred)}')
    # 7.绘图
    # plt.scatter(x, y, label='真实值')
    # plt.plot(np.sort(x), y_pred[np.argsort(x)], label='预测值') # sort() 对x轴进行排序 默认升序 argsort() 返回排序后的索引
    x_flat = x.ravel()  # 把二维数组展平成一维 (100, 1) → (100,)
    sorted_indices = np.argsort(x_flat)  # 获取排序后的索引
    plt.scatter(x_flat, y, label='真实值')
    plt.plot(x_flat[sorted_indices], y_pred[sorted_indices], label='预测值')
    plt.legend()
    plt.show()
# 5.定义函数，模拟：L2正则化
def dm05_l2_regression():
    # 1.准备数据集（正好拟合）
    # 指定随机种子，则每次生成的数据都是固定的
    np.random.seed(23)
    # 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, size=100)  # 参1：最小值 参2：最大值 参3：样本数量
    # 基于x轴值，通过线性公式，生成y轴 100个数据，模拟：目标值 y = kx + b = 0.5*x**2 + x + 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2.数据预处理，把x轴（特征）转成 多行一列的形式
    x = x.reshape(-1, 1)  # 转为单列
    # 因为目前特征列只有1列，模型过于简单，为了模拟过拟合，我们增加9列 从而增加模型复杂度
    x3 = np.hstack([x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10])  # 该函数的作用：将多个数组按列方向拼接
    # print(f'处理后的数据：{x}')

    # 3.特征工程，这里不做了，直接用100条数据，先训练后预测

    # 4.模型训练
    # 创建L2正则化对象
    estimator = Ridge(alpha=0.1)  # alpha：惩罚系数 默认：0.1
    # 模型训练
    estimator.fit(x3, y)
    # 5.模型预测
    y_pred = estimator.predict(x3)
    # 6.模型评估
    print(f'均方误差为：{mean_squared_error(y, y_pred)}')
    # 7.绘图
    # plt.scatter(x, y, label='真实值')
    # plt.plot(np.sort(x), y_pred[np.argsort(x)], label='预测值') # sort() 对x轴进行排序 默认升序 argsort() 返回排序后的索引
    x_flat = x.ravel()  # 把二维数组展平成一维 (100, 1) → (100,)
    sorted_indices = np.argsort(x_flat)  # 获取排序后的索引
    plt.scatter(x_flat, y, label='真实值')
    plt.plot(x_flat[sorted_indices], y_pred[sorted_indices], label='预测值')
    plt.legend()
    plt.show()
# 6.测试
if __name__ == '__main__':
    # dm01_under_fitting()
    # dm02_just_fitting()
    # dm03_over_fitting()
    # dm04_l1_regression()
    dm05_l2_regression()
