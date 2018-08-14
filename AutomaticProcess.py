# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:09:16 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
#选择前最好的几种特征
from sklearn.feature_selection import SelectKBest


'''
数据准备和生成模型的Pipeline
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

steps=[]
steps.append(('Standardize',StandardScaler()))
steps.append(('lda',LinearDiscriminantAnalysis()))
model=Pipeline(steps)
result=cross_val_score(model,X,y,cv=kfold)
print('Pipeline LDA:',result.mean())


'''
使用Pipeline特征选择
'''
array=data.values
X=array[:,0:8]
y=array[:,8]
num_folds=10
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)

#生成FeatureUnion
features=[]
features.append(('pca',PCA()))
#选择前最好的6种特征
features.append(('select_best',SelectKBest(k=6)))

#生成Pipeline
steps=[]
steps.append(('feature_union',FeatureUnion(features)))
steps.append(('logistic',LogisticRegression()))
model=Pipeline(steps)
result=cross_val_score(model,X,y,cv=kfold)
print('使用Pipeline特征选择',result.mean())








