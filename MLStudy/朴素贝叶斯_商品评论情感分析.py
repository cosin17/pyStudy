"""
案例：
    演示通过 朴素贝叶斯算法 实现 商品评论情感分析，即：好评，差评

朴素贝叶斯介绍：
    概述：
        贝叶斯：仅仅依赖 概率 就可以进行分类的 一种机器学习算法。
        朴素：不考虑特征之间的关联性，即：每个特征之间是独立的。
            原始：P(AB) = P(A) * P(B|A) = P(B) * P(A|B)
            加入朴素后：P(AB) = P(A) * P(B)
        细节：
            因为我们分词要用到jieba分词器，记得先安装jieba分词器。
"""

# 导包
import numpy as np                # 数据计算包
import pandas as pd               # 数据处理包
import matplotlib.pyplot as plt   # 数据可视化包
import jieba                      # 中文分词器
from sklearn.feature_extraction.text import CountVectorizer # 词频统计包，把评论内容 转成 词频矩阵
from sklearn.naive_bayes import MultinomialNB               # 朴素贝叶斯对象
from sklearn.metrics import accuracy_score

# 1.读取文件，获取到原数据
df = pd.read_csv('./data/书籍评价.csv',encoding='gbk')
# df.info()

# 2.数据预处理
# 2.1添加labels列，充当：标签列，好评 ——> 1，差评 ——> 0
df['labels'] = np.where(df['评价']=='好评',1,0)

# 2.2 抽取labels列，充当：标签列
y = df['labels']

# 2.3 演示jieba 分词
# print(jieba.lcut('好好学习，天天向上！我爱你你爱我，蜜雪冰城甜蜜蜜！'))

# 2.4 对用户的评论信息，做切词
# 数据格式：[[第1条评论切词1，切词2，...],[第2条评论切词1，切词2，...],...]
# comment_list = [jieba.lcut(line) for line in df['内容']]
# 数据格式：['第1条评论切词1，切词2，...','第2条评论切词1，切词2，...','...']
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]  # join() 方法：把列表中的元素，用指定的字符连接起来
print(comment_list)

# 2.5 加载 停用词列表，即：里边记录的词，不需要与模型训练，预测，要被删除的词，例如：的，他，她，啊，哈...
with open('./data/stopwords.txt','r',encoding='utf-8') as src_f:
    stopwords_list = src_f.readlines() # 读取文件中的所有行，返回一个列表
    stopwords_list = [line.strip() for line in stopwords_list] # 去掉每个元素的首尾空格
    stopwords_list = list(set(stopwords_list)) # set() 去重 list() 转换为列表类型

# 2.6 创建向量化对象，从 评论切词列表(comment_list) 中 删除 停用词，并且统计词频(单词矩阵)
transfer = CountVectorizer(stop_words=stopwords_list)
# 2.7 统计词频矩阵,先训练，后转换，在转数组
transfer.fit(comment_list)
x = transfer.transform(comment_list).toarray()
print(x)

# 2.8 看一下 我们13条评论，切词，且删除 停用词后，一共剩下多少个词了
print(transfer.get_feature_names_out()) # 所有特征的名称
print(len(transfer.get_feature_names_out())) # 词频矩阵的列数

# 2.9 因为就 13 条数据，我们把前10条当训练集，后三条当测试集
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

# 3.特征工程

# 4.模型训练
estimator = MultinomialNB().fit(x_train, y_train)

# 5.模型预测
y_pred = estimator.predict(x_test)
print(f'模型预测结果：{y_pred}')

# 6.模型评估
print(f'准确率：{accuracy_score(y_test, y_pred)}')
