"""
案例：
    通过逻辑回归算法，对于电信用户数据建模，进行流失预测分析
"""


# 导包
import pandas as pd
import numpy as np
import seaborn as sns # 数据可视化库
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report # 准确确率 精确率 召回率 F1值 分类报告


# 1.定义函数 演示：数据的预处理
def dmo1_data_preprocess():
    # 1.读取CSV文件，获取到df对象
    churn_df = pd.read_csv('./data/churn.csv')

    # 2.查看数据集(处理前)
    # churn_df.info()
    # print(churn_df.head(5))
    # 3.因为churn 和 gender列是字符串，所以需要进行one-hot编码(热编码处理)
    churn_df = pd.get_dummies(churn_df,columns=['churn','gender'])

    # 4.查看数据集(处理后)
    # churn_df.info()
    # print(churn_df.head(5))

    # 5.删除one-hot处理后，冗余的列
    # axis=1 表示删除列 inplace=True 表示在原数据上进行修改
    churn_df.drop(['churn_No','gender_Male'],axis=1,inplace=True)

    # 6.修改列名，将churn.Yes ——> flag，充当标签列
    churn_df.rename(columns={'churn_Yes':'flag'},inplace=True)
    churn_df.info()
    print(churn_df.head(5))  # False ——> 不流失 True ——> 流失

    # 7.查看数值的分布
    print(churn_df['flag'].value_counts()) # False 5174 True 1869

# 2.定义函数 演示：数据的可视化
def dmo2_data_visualization():
    # 1.读取CSV文件，获取到df对象
    churn_df = pd.read_csv('./data/churn.csv')
    # 2.对object类型的列做 one-hot 编码处理
    churn_df = pd.get_dummies(churn_df,columns=['churn','gender'])
    # 3.删除one-hot处理后，冗余的列
    # axis=1 表示删除列 inplace=True 表示在原数据上进行修改
    churn_df.drop(['churn_No','gender_Male'],axis=1,inplace=True)
    # 4.修改列名，将churn.Yes ——> flag，充当标签列
    churn_df.rename(columns={'churn_Yes':'flag'},inplace=True)
    """
    Index(['Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Female'],
      dtype='str')
    """
    # 6.查看列名，方便我们一会儿抽取 特征
    print(churn_df.columns)
    # 7.数据的可视化，绘制 计数柱状图
    sns.countplot(churn_df,x='Contract_Month',hue='flag') # hue 表示根据flag列进行分组绘制
    plt.show()

# 3.定义函数 演示：逻辑回归算法的模型训练，预测，评估
def dm03_logistic_regression():
    # 1.加载数据集
    churn_df = pd.read_csv('./data/churn.csv')

    # 2.数据的预处理
    # 2.1对object类型的列做 one-hot 编码处理
    churn_df = pd.get_dummies(churn_df,columns=['churn','gender'])
    # 2.2删除one-hot处理后，冗余的列
    churn_df.drop(['churn_No','gender_Male'],axis=1,inplace=True)
    # 2.3修改列名，将churn.Yes ——> flag，充当标签列
    churn_df.rename(columns={'churn_Yes':'flag'},inplace=True)
    # 2.4提取特征列 和 标签列
    # x的特征列：月度会员 是否有互联网服务 是否使用电子支付方式
    x = churn_df[['Contract_Month','internet_att','PaymentElectronic']]
    y = churn_df['flag']
    # 2.5将数据集进行训练集和测试集的划分
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=23)

    # 3.特征工程
    # 4.模型训练
    # 4.1创建逻辑回归模型对象
    estimator = LogisticRegression()
    # 4.2模型训练
    estimator.fit(x_train,y_train)

    # 5.模型预测
    y_pred = estimator.predict(x_test)
    print(f'模型预测结果：{y_pred}')

    # 6.模型评估
    print(f'预测前准确率：{estimator.score(x_test,y_test)}')
    print(f'预测后准确率：{accuracy_score(y_test,y_pred)}')
    print(f'精确率：{precision_score(y_test,y_pred)}')
    print(f'召回率：{recall_score(y_test,y_pred)}')
    print(f'F1值：{f1_score(y_test,y_pred)}')
    print(f'分类报告：{classification_report(y_test,y_pred)}')


if __name__ == '__main__':
    # dmo1_data_preprocess()
    # dmo2_data_visualization()
    dm03_logistic_regression()
