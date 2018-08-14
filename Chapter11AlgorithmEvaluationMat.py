# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:40:11 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]


'''
k折分类准确度
'''
#初始化K的次数
num_folds=10
#随机种子
seed=7
#初始化Kflod
kfold=KFold(n_splits=num_folds,random_state=seed)
#创建LogisticRegression模型
model=LogisticRegression()
#交叉验证(此处进行模型的训练和评估)
result=cross_val_score(model,X,y,cv=kfold)
print('k折分类准确度算法评估准确度为:{0}，标准差为:{1}'.format(result.mean(),result.std()))



'''
对数损失函数
'''
n_splits=10
seed=7
kfold=KFold(n_splits=n_splits,random_state=seed)
model=LogisticRegression()
#定义对数损失函数标签
scoring='neg_log_loss'
result=cross_val_score(model,X,y,cv=kfold,scoring=scoring)

print('对数损失函数算法评估准确度为:{0}，标准差为:{1}'.format(result.mean(),result.std()))


'''
AUC图
'''
n_splits=10
seed=7
kfold=KFold(n_splits=n_splits,random_state=seed)
model=LogisticRegression()
scoring='roc_auc'
result=cross_val_score(model,X,y,scoring=scoring,cv=kfold)
print('AUC图算法评估准确度为:{0}，标准差为:{1}'.format(result.mean(),result.std()))


'''
混淆矩阵
'''
test_size=0.33
seed=49
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=seed,test_size=test_size)
model=LogisticRegression()
model.fit(X_train,y_train)
predicted=model.predict(X_test)
#构建混淆矩阵
matrix=confusion_matrix(y_test,predicted)
classes=['0','1']
dataframe=pd.DataFrame(data=matrix,index=classes,columns=classes)
print(dataframe)


'''
分类报告
'''
test_size=0.33
seed=49
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=seed,test_size=test_size)
model=LogisticRegression()
model.fit(X_train,y_train)
predicted=model.predict(X_test)
report=classification_report(y_test,predicted)
print(report)


'''
回归算法矩阵:波士顿房价数据集
'''


'''
平均绝对误差
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=LinearRegression()
scoring='neg_mean_absolute_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print('MAE算法评估准确度为:{0}，标准差为:{1}'.format(result.mean(),result.std()))


'''
均方误差
'''
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=LinearRegression()
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print('MSE算法评估准确度为:{0}，标准差为:{1}'.format(result.mean(),result.std()))



'''
决定系数R^2
'''
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=LinearRegression()
scoring='r2'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print('r2算法评估准确度为:{0}，标准差为:{1}'.format(result.mean(),result.std()))







