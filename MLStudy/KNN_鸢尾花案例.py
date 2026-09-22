# 导入鸢尾花数据集
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 导入训练集测试集划分函数
from sklearn.preprocessing import StandardScaler # 导入标准化类
from sklearn.neighbors import KNeighborsClassifier # 导入KNN分类器
from sklearn.metrics import accuracy_score # 导入准确率评估指标


# 1.定义函数，加载鸢尾花数据集，比查看数据集
def dm01_load_iris():
    iris_data = load_iris()
    # print(f'数据集：{iris_data}') # 字典形态
    print(f'数据集类型：{type(iris_data)}') # <class 'sklearn.datasets._base.Bunch'>
    print(f'特征名称：{iris_data.feature_names}')
    print(f'数据所有的键：{iris_data.keys()}')
    print(f'数据集的框架：{iris_data.frame}') # None

# 2.定义函数，绘制数据集的散点图（seaborn）
def dm02_show_iris():
    # 加载数据集
    iris_data = load_iris()
    # 把 鸢尾花数据集封装成DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names) # columns 表示列名
    # 给df对象新增1列 ——> 标签列
    iris_df['label'] = iris_data.target

    # 通过 Seaborn 绘制数据集的散点图
    sns.lmplot(data = iris_df, x = 'sepal length (cm)', y = 'sepal width (cm)', hue = 'label',fit_reg = True) # hue 表示根据标签列进行分类，fit_reg 表示不拟合回归线

    # 设置标题 显式
    plt.title('iris data')
    plt.tight_layout() # 自动调整子图参数，使子图之间的间距更小
    print(iris_df)
    plt.show()

# dm01_load_iris()
dm02_show_iris()