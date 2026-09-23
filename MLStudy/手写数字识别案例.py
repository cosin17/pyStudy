"""
案例：演示 KNN算法 手写数字识别案例

介绍：
    每张图片都是由 28*28 像素组成，每个像素的取值范围是 0-255
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib # 保存模型
from collections import Counter # 统计每个数字出现的次数

# 定义函数，接收用户传入的索引，返回对应的图片
def show_digit(idx):
    # 读取数据集
    df = pd.read_csv('./data/train.csv')
    print(df) # 4200*785

    # 判断传入的索引是否越界
    if idx < 0 or idx > len(df)-1:
        print('索引越界!')
        return

    # 走到这里说明没有越界，正常获取数据
    x = df.iloc[:,1:] # 所有行，从第 1 列开始到最后（左闭右开）
    y = df.iloc[:,0] # 所有行，只取第 0 列

    # 查看用户传入的索引对应的图片是几？
    print(f'用户传入的索引对应的图片数字是：{y.iloc[idx]}')

    # 查看 用户传入的索引对应的图片 的形状
    print(f'用户传入的索引对应的图片形状是：{x.iloc[idx].shape}')
    print(x.iloc[idx].values) # 具体的784个像素点数据

    # 把 784 个像素点数据转换为 28*28 的图片
    x_img = x.iloc[idx].values.reshape(28,28)

    # 具体的绘制灰度图的动作
    plt.imshow(x_img,cmap='gray') # imshow 能把这堆像素映射成颜色
    plt.axis('off') # 关闭坐标轴
    plt.show()

# 定义函数 训练模型，并保存模
def train_model():
    # 读取数据集
    df = pd.read_csv('./data/train.csv')
    # 数据的预处理，拆分训练集和测试集
    x = df.iloc[:,1:] # 特征列
    y = df.iloc[:,0] # 标签列

    # 对特征列拆分前进行 归一化
    x = x / 255
    # 拆分训练集和测试集，比例是 80%20%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23,stratify=y) # stratify=y 表示拆分时参考y值进行抽取

    # 模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 模型评估
    print(f'准确率：{estimator.score(x_test, y_test)}')
    print(f'准确率：{accuracy_score(y_test, estimator.predict(x_test))}')

    # 保存模型
    # 参1：模型对象 参2：模型保存的路径
    joblib.dump(estimator, './model/手写数字识别.pkl') # pickle文件：Python(Pandas)独有的文件类型
    print('模型保存成功!')
# 测试
if __name__ == '__main__':
    # show_digit(9)
    train_model()
