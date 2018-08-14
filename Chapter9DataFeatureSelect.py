# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:02:58 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier



#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]

'''
单变量特征选择
'''
#特征选择
#选择特征通过k个最高分数,score_func使用卡方检验
test=SelectKBest(score_func=chi2,k=4)
fit=test.fit(X,y)
np.set_printoptions(precision=3)
print(fit.scores_)
features=fit.transform(X)
print(features)
'''
[ 110.727 1406.59    17.505   51.008 2219.398  127.671    5.356  178.011]
[[ 85.    0.   26.6  31. ]
 [183.    0.   23.3  32. ]
 [ 89.   94.   28.1  21. ]
 ...
 [121.  112.   26.2  30. ]
 [126.    0.   30.1  47. ]
 [ 93.    0.   30.4  23. ]]
'''


'''
递归特征消除：
使用一个基模型来进行多轮训练，
每轮训练后消除若干权值系数的特征，
再基于新的特征集进行下一轮训练
'''
#特征选择
model=LogisticRegression()
#使用LogisticRegression来进行多轮训练
rfe=RFE(model,3)
fit=rfe.fit(X,y)
print('特征个数:{0}'.format(fit.n_features_))
print('被选定的特征个数:{0}'.format(fit.support_))
print('特征排名:{0}'.format(fit.ranking_))


'''
PCA主成分分析（无监督）
LDA（有监督）
'''
#特征选定
#选定3个主要特征
pca=PCA(n_components=3)
fit=pca.fit(X)
print('解释方差:%s'%fit.explained_variance_ratio_)
print(fit.components_)

'''
特征重要性
通过决策树计算特征的重要性
'''
#特征选择
model=ExtraTreesClassifier()
fit=model.fit(X,y)
print(fit.feature_importances_)
#[0.107 0.225 0.1   0.086 0.069 0.148 0.123 0.141]














