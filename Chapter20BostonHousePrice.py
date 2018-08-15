# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:36:53 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

#导入数据
datasets=load_boston()
X=datasets.data
y=datasets.target

#划分训练、测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

print(X.shape)
print(y.shape)
#(506, 13)
#(506,)

#使用评估算法
#10折交叉验证
num_folds=10
seed=7
#均方差误差
scoring='neg_mean_squared_error'


'''
使用3个线性算法和3个非线性算法
LR LASSO EN
CART SVM KNN
'''
#评估算法
models={}
models['LR']=LinearRegression()
models['LASSO']=Lasso()
models['EN']=ElasticNet()
models['KNN']=KNeighborsRegressor()
models['SVR']=SVR()
models['CART']=DecisionTreeRegressor()

#获取算法准确度
results=[]
for key in models:
    #K折交叉验证
    kfold=KFold(n_splits=num_folds,random_state=seed)
    #验证结果
    cv_result=cross_val_score(models[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))

'''
查看10这交叉分离验证的结果
箱线图
'''
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


'''
正态化数据
'''
#使用pipeline
pipelines={}
pipelines['ScalerLR']=Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
pipelines['ScalerLasso']=Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])
pipelines['ScalerEN']=Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])
pipelines['ScalerSVR']=Pipeline([('Scaler',StandardScaler()),('SVR',SVR())])
pipelines['ScalerKNN']=Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])
pipelines['ScalerCART']=Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])

results=[]
for key in pipelines:
    kfold=KFold(random_state=seed,n_splits=num_folds)
    cv_result=cross_val_score(pipelines[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('机器学习单一算法%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))

'''
查看10这交叉分离验证的结果
箱线图
'''
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


'''
从上观察
KNN是最好的算法

因此主要对KNN进行参数调整
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
#调整参数
param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
#构建模型
model=KNeighborsRegressor()
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
接下来使用集成算法
装袋算法：
RF   ET
提升算法：
AB  GBM
'''
ensembles={}
ensembles['ScaledAB']=Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])
ensembles['ScaledAB-KNN']=Pipeline([('Scaler',StandardScaler()),('ABKNN',AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
ensembles['ScaledAB-LR']=Pipeline([('Scaler',StandardScaler()),('ABLR',AdaBoostRegressor(LinearRegression()))])
ensembles['ScaledRFR']=Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestRegressor())])
ensembles['ScaledETR']=Pipeline([('Scaler',StandardScaler()),('ETR',ExtraTreesRegressor())])
ensembles['ScaledGBR']=Pipeline([('Scaler',StandardScaler()),('GBR',GradientBoostingRegressor())])

results=[]
for key in ensembles:
    kfold=KFold(random_state=seed,n_splits=num_folds)
    cv_result=cross_val_score(ensembles[key],X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('集成算法%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))



'''
查看10这交叉分离验证的结果
箱线图
'''
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()


'''
对GBM和ET算法进行调参
'''
'''
GBM
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
#调整参数
param_grid={'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
#构建模型
model=GradientBoostingRegressor()
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
ET
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
#调整参数
param_grid={'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
#构建模型
model=ExtraTreesRegressor()
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
    print('ET%f (%f) with %r'%(mean,std,param))




'''
确定最终模型
ET
'''
#正态化数据
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
gbr=ExtraTreesRegressor(n_estimators=80)
gbr.fit(X=rescaledX,y=y_train)

#评估算法模型
rescaledX_validation=scaler.transform(X_test)
preditions=gbr.predict(X_test)
print('准确度:',mean_squared_error(preditions,y_test))
