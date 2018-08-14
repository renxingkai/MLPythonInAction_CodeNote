# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:39:37 2018

@author: Administrator
"""

'''
算法调参
两种自动寻找最优化参数的算法：
网格搜索优化参数
随机搜索优化参数
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

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

'''
网格搜索优化参数
通过遍历已定义参数的列表，来评估算法的参数，从而找到最优参数
适用于3、4个超参数（或更少）
'''
#创建模型
model=Ridge()
#设置要遍历的参数
param_grid={'alpha':[1,0.1,0.01,0.001,0]}
#通过网格搜索查询最优参数
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(X,y)
#搜索结果
print('最高得分：%.3f'%grid.best_score_)
print('最优参数:%s'%grid.best_params_)

'''
随即搜索优化参数
适用于参数较多的情况
适用固定次数的迭代，采用随机采样分布的方式搜索合适的参数
'''
#创建模型
model=Ridge()
#设置要遍历的参数,(0,1)之间均匀分布的参数
param_grid={'alpha':uniform()}
#通过随机搜索查询最优参数,迭代100次
grid=RandomizedSearchCV(n_iter=100,estimator=model,param_distributions=param_grid,random_state=seed)
grid.fit(X,y)
#搜索结果
print('最高得分：%.3f'%grid.best_score_)
print('最优参数:%s'%grid.best_params_)






























