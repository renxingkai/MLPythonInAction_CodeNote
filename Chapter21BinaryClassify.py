# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:48:15 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#导入数据
filename='sonar.csv'
dataset=pd.read_csv(filename)

#分析数据的维度
print(dataset.shape)
#(207, 61) 207行数据，60个特征，1个label

#查看数据的类型
pd.set_option('display.max_rows',500)
#print(dataset.dtypes)

#查看最开始的20条数据
pd.set_option('display.width',100)
#print(dataset.head(20))

#描述数据统计信息
pd.set_option('precision',3)
#print(dataset.describe())

#查看数据的分类分布
#print(dataset.groupby('60').size())


'''
数据可视化
'''
#直方图
#dataset.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
#plt.show()

#密度分布图
#dataset.plot(kind='density',subplots=True,layout=(8,8),sharex=False,legend=False,fontsize=1)
#plt.show()

#关系矩阵图
#fig=plt.figure()
#ax=fig.add_subplot(111)
#cax=ax.matshow(dataset.corr(),vmin=-1,vmax=1,interpolation='none')
#fig.colorbar(cax)
#plt.show()

'''
分离评估数据集
'''
array=dataset.values
X=array[:,0:60].astype(float)
y=array[:,60]
validation_size=0.2
seed=7
X_train,X_validation,y_train,y_validation=train_test_split(X,y,test_size=validation_size,random_state=seed)

'''
评估算法
'''
#评估算法的基准
num_folds=10
seed=7
scoring='accuracy'

'''
使用6种不同算法进行评估
LR LDA
CART SVM NB KNN
'''
models={}
models['LR']=LinearRegression()
models['LDA']=LinearDiscriminantAnalysis()
models['KNN']=KNeighborsClassifier()
models['CART']=DecisionTreeClassifier()
models['NB']=GaussianNB()
models['SVM']=SVC()

#比较各算法的准确度
results=[]
for key in models:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_results=cross_val_score(models[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    print('%s准确度:%f ,方差:%f'%(models[key],cv_results.mean(),cv_results.std()))


#评估算法---箱线图
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys)
plt.show()

'''
正态化数据
'''
#使用pipeline
pipelines={}
pipelines['ScalerLR']=Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
pipelines['ScalerLDA']=Pipeline([('Scaler',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])
pipelines['ScalerKNN']=Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])
pipelines['ScalerCART']=Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeClassifier())])
pipelines['ScalerNB']=Pipeline([('Scaler',StandardScaler()),('NB',GaussianNB())])
pipelines['ScalerSVM']=Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])

results=[]
for key in pipelines:
    kfold=KFold(random_state=seed,n_splits=num_folds)
    cv_result=cross_val_score(pipelines[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('机器学习单一算法%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))



#评估算法---箱线图
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(pipelines.keys)
plt.show()


'''
主要对KNN进行参数调整
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
#调整参数
param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
#构建模型
model=KNeighborsClassifier()
#10折交叉验证
kfold=KFold(n_splits=num_folds,random_state=seed)
#网格搜索
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
#获取结果
grid_result=grid.fit(X=rescaledX,y=y_train)
print('最优参数:%s,获取分数:%s'%(grid_result.best_params_,grid_result.best_score_))
#zip合并矩阵
cv_results=zip(grid_result.cv_results_['mean_test_score'],
               grid_result.cv_results_['std_test_score'],
               grid_result.cv_results_['params'])

for mean,std,param in cv_results:
    print('%f (%f) with %r'%(mean,std,param))



'''
主要对SVM进行参数调整
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train).astype(float)
#调整参数
param_grid={}
param_grid['C']={0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0}
param_grid['kernel']={'linear','poly','rbf','sigmoid','precomputed'}
#构建模型
model=SVC()
#10折交叉验证
kfold=KFold(n_splits=num_folds,random_state=seed)
#网格搜索
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
#获取结果
grid_result=grid.fit(X=rescaledX,y=y_train)
print('最优参数:%s,获取分数:%s'%(grid_result.best_params_,grid_result.best_score_))
#zip合并矩阵
cv_results=zip(grid_result.cv_results_['mean_test_score'],
               grid_result.cv_results_['std_test_score'],
               grid_result.cv_results_['params'])

for mean,std,param in cv_results:
    print('%f (%f) with %r'%(mean,std,param))


'''
集成算法
RF ET
AB GBM
'''
#使用pipeline
ensembles={}
ensembles['ScalerAB']=Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostClassifier())])
ensembles['ScalerGBM']=Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingClassifier())])
ensembles['ScalerRFR']=Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestClassifier())])
ensembles['ScalerET']=Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesClassifier())])

results=[]
for key in pipelines:
    kfold=KFold(random_state=seed,n_splits=num_folds)
    cv_result=cross_val_score(ensembles[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('机器学习集成算法%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))



#评估算法---箱线图
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys)
plt.show()


'''
GBM算法调参
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
#调整参数
param_grid={'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
#构建模型
model=GradientBoostingClassifier()
#10折交叉验证
kfold=KFold(n_splits=num_folds,random_state=seed)
#网格搜索
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
#获取结果
grid_result=grid.fit(X=rescaledX,y=y_train)
print('最优参数:%s,获取分数:%s'%(grid_result.best_params_,grid_result.best_score_))
#zip合并矩阵
cv_results=zip(grid_result.cv_results_['mean_test_score'],
               grid_result.cv_results_['std_test_score'],
               grid_result.cv_results_['params'])

for mean,std,param in cv_results:
    print('GBM%f (%f) with %r'%(mean,std,param))


'''
确定最终模型
ET
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
model=SVC(C=1.0,kernel='rbf')
model.fit(X=rescaledX,y=y_train)

#评估算法模型
rescaledX_validation=scaler.transform(X_validation)
preditions=model.predict(X_validation)
print('准确度:',accuracy_score(preditions,y_validation))
print('混淆矩阵:',confusion_matrix(preditions,y_validation))
print('分类结果:',classification_report(preditions,y_validation))