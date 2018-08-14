# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:31:32 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit



#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]


'''
划分训练集和测试集
'''
#划分测试集大小为33%
test_size=0.33
#数据随机的粒度，通过指定数据随机的粒度，可以确保每次执行程序得到相同的结果
#有助于比较两个不同的算法生成的模型的结果
seed=4
#划分训练集、测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=seed,test_size=test_size)
#构建模型
model=LogisticRegression()
#训练模型
model.fit(X_train,y_train)
#训练集进行打分
print(model.score(X_train,y_train))
#测试集打分
print(model.score(X_test,y_test))



'''
K折交叉验证
'''
#交叉验证分组数，分为10组
num_folds=10
#随机种子选择7
seed=7

#构造交叉验证
kfold=KFold(n_splits=num_folds,random_state=seed)
#交叉验证结果
result=cross_val_score(model,X,y,cv=kfold)
print('算法评估结果:%.3f%%(方差为:%.3f%%)'%(result.mean()*100,result.std()*100))


'''
弃一交叉验证分离(计算成本较高)
'''
loocv=LeaveOneOut()
result=cross_val_score(model,X,y,cv=loocv)
print('算法评估结果:%.3f%%(方差为:%.3f%%)'%(result.mean()*100,result.std()*100))



'''
重复随机分离评估数据集与训练数据集
'''
#将重复随机分离数据集和训练集重复10次
n_splits=10
#测试集比例
test_size=0.33
#随机种子
seed=7
#交叉验证
kfold=ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=seed)
#结果
result=cross_val_score(model,X,y,cv=kfold)
print('算法评估结果:%.3f%%(方差为:%.3f%%)'%(result.mean()*100,result.std()*100))












































