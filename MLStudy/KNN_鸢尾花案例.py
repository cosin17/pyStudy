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

# 3.定义函数 切分训练集和测试集
def dm03_split_train_test():
    iris_data = load_iris()
    # 数据预处理: 从150个特征和标签中，按照 8：2 的比例，切分训练集和测试集
    # 返回值：训练集和测试集的特征和标签
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)

    # 打印训练集和测试集的特征和标签
    print(f'训练集特征：{x_train},个数：{len(x_train)}') # 120条，每条4列
    print(f'训练集标签：{y_train},个数：{len(y_train)}') # 120条 ，每条1列
    print(f'测试集特征：{x_test},个数：{len(x_test)}') # 30条，每条4列
    print(f'测试集标签：{y_test},个数：{len(y_test)}') # 30条，每条1列76

# 4.定义函数，实现鸢尾花完整案例——> 加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测。
def dm04_iris_evaluate_test():
    # 加载数据集
    iris_data = load_iris()
    # 数据预处理: 从150个特征和标签中，按照 8：2 的比例，切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)
    # 特征工程
    # 思考1：特征提取：因为源数据只有4个特征列，且都是我们需要的，所有不需要进行特征提取
    # 思考2：特征预处理：因为原数据的4列特征差值不大，所以我们无需做特征预处理，但是加入特征预处理会让我们的代码更完善

    # 创建标准化对象
    transfer = StandardScaler()
    # 对特征例进行标准化，即：x_train:训练集的特征数据，x_test:测试集的特征数据
    # fit_transform：兼具fit和transform的功能，即训练、转换，适用于第一次进行标准化的时候使用
    x_train = transfer.fit_transform(x_train)
    # transform:只有转换，适用于重复进行标准化动作时使用，一般用于对测试集进行标准化
    x_test = transfer.transform(x_test)

    # 模型训练
    # 创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3) # 分类思路 默认n_neighbors=5
    # 对模型进行训练
    estimator.fit(x_train, y_train)

    # 模型预测
    # 场景1：对刚才切分的 测试集(30条) 进行测试
    y_pred = estimator.predict(x_test)
    print(f'预测结果为：{y_pred}')
    # 场景2：对新的数据集 进行测试
    my_data = [[7.8,2.1,3.9,1.6]]
    # 对新的数据集 进行标准化
    my_data = transfer.transform(my_data)
    # 模型预测
    y_pred_new = estimator.predict(my_data)
    print(f'新的数据集 预测结果为：{y_pred_new}')
    print(f'新的数据集 预测概率为：{estimator.predict_proba(my_data)}')

    # 模型评估
    # 方式1：直接评分，基于：训练集的特征 和 训练集的标签
    print(f'训练集准确率为：{estimator.score(x_train, y_train)}')
    # 方式2：基于测试集的标签 和 预测结果 进行评分
    print(f'测试集准确率为：{accuracy_score(y_test, y_pred)}')

# dm01_load_iris()
# dm02_show_iris()
dm04_iris_evaluate_test()
