# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:40:55 2018

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

'''
......审查分类算法.......
'''

'''
逻辑回归
'''
#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=LogisticRegression()
result=cross_val_score(model,X,y,cv=kfold)
print("逻辑回归",result.mean())

'''
LDA算法
'''
#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=LinearDiscriminantAnalysis()
result=cross_val_score(model,X,y,cv=kfold)
print("LDA算法",result.mean())



'''
非线性算法
'''

'''
KNN
'''
#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=KNeighborsClassifier()
result=cross_val_score(model,X,y,cv=kfold)
print("KNN算法",result.mean())

'''
Bayes
'''
#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=GaussianNB()
result=cross_val_score(model,X,y,cv=kfold)
print("Bayes算法",result.mean())

'''
DecisionTree
'''
#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=DecisionTreeClassifier()
result=cross_val_score(model,X,y,cv=kfold)
print("DecisionTree算法",result.mean())

'''
SVM
'''
#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=SVC()
result=cross_val_score(model,X,y,cv=kfold)
print("SVM算法",result.mean())



'''
......审查回归算法.......
波士顿房价数据集
'''

'''
线性回归算法
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=LinearRegression()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("线性回归算法",result.mean())


'''
岭回归
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=Ridge()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("岭回归算法",result.mean())


'''
Lasso回归
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=Lasso()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("Lasso回归算法",result.mean())


'''
弹性网络回归
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=ElasticNet()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("弹性网络回归算法",result.mean())


'''
KNN回归
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=KNeighborsRegressor()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("KNN回归算法",result.mean())


'''
回归树
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=DecisionTreeRegressor()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("回归树算法",result.mean())

'''
SVM回归
'''
datasets=load_boston()
X=datasets.data
y=datasets.target
n_splits=10
seed=7
kflod=KFold(n_splits=n_splits,random_state=seed)
model=SVR()
#均方误差
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,y,scoring=scoring,cv=kflod)
print("SVM回归算法",result.mean())




'''
机器学习算法的比较
糖尿病数据
'''
#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
models={}
models['LR']=LogisticRegression()
models['LDA']=LinearDiscriminantAnalysis()
models['KNN']=KNeighborsClassifier()
models['CART']=DecisionTreeClassifier()
models['SVM']=SVC()
models['NB']=GaussianNB()

results=[]
for name in models:
    result=cross_val_score(models[name],X,y,cv=kfold)
    results.append(result)
    msg='%s:%.3f(%.3f)'%(name,result.mean(),result.std())
    print(msg)
    
#图表显示
fig=plt.figure()
#标题
fig.suptitle('Algorithm Comparison')
#画出子图，参数一：子图总行数，参数二：子图总列数；参数三：子图位置
ax=fig.add_subplot(111)
#画出箱线图
plt.boxplot(results)
#设置x轴名称
ax.set_xticklabels(models.keys())
plt.show()




