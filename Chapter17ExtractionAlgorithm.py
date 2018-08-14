# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:34:10 2018

@author: Administrator
"""

'''
集成算法：
将多种机器学习算法组合起来，使计算出来的结果更好
'''

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

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
装袋算法(Bagging)：............
先将训练集分离成多个子集，然后通过各个子集训练多个模型
装袋算法在具有较大方差时比较有用
使用糖尿病数据
'''

'''
装袋决策树(Bagged Decision Trees)
'''
cart=DecisionTreeClassifier()
#构建100棵决策树
num_tree=100
#构建装袋决策树模型
model=BaggingClassifier(base_estimator=cart,n_estimators=num_tree,random_state=seed)
result=cross_val_score(model,X,y,cv=kfold)
print('装袋决策树(Bagged Decision Trees)',result.mean())


'''
随机森林(Random Forest)
'''
#构建100棵树的森林
num_trees=100
#最大特征设为3
max_features=3
#构建模型
model=RandomForestClassifier(n_estimators=num_trees,random_state=seed,max_features=max_features)
result=cross_val_score(model,X,y,cv=kfold)
print('随机森林(Random Forest)',result.mean())



'''
极端随机数(Extra Trees)
使用所有的样本去构建每棵决策树
'''
#构建100棵树的森林
num_trees=100
#最大特征设为7，糖尿病数据的数据所有特征
max_features=7
#构建模型
model=ExtraTreesClassifier(n_estimators=num_trees,random_state=seed,max_features=max_features)
result=cross_val_score(model,X,y,cv=kfold)
print('极端随机数(Extra Trees)',result.mean())


'''
提升算法(Boosting)：............
训练多个模型并组成一个序列，序列中的每一个模型都会去修正前一个模型的错误
使用糖尿病数据
'''

'''
AdaBoost算法
'''
#30棵树模型
num_trees=30
model=AdaBoostClassifier(n_estimators=num_trees,random_state=seed)
result=cross_val_score(model,X,y,cv=kfold)
print('AdaBoost算法',result.mean())

'''
随机梯度提升算法(SGB)Stohastic Gradient Boosting
'''
#构建100棵树
num_trees=100
model=GradientBoostingClassifier(n_estimators=num_trees,random_state=seed)
result=cross_val_score(model,X,y,cv=kfold)
print('随机梯度提升算法(SGB)Stohastic Gradient Boosting',result.mean())


'''
投票算法(Voting)：............
训练多个模型，并采用样本统计来提高模型的准确度
通过创建两个或者多个算法模型，利用投票算法将这些算法包装起来，计算各个子模型的平均预测状况
使用糖尿病数据
'''
#决策树
models=[]
model_logistic=LogisticRegression()
models.append(('logistic',model_logistic))
model_cart=DecisionTreeClassifier()
models.append(('cart',model_cart))
model_svc=SVC()
models.append(('SVC',model_svc))
ensemble_model=VotingClassifier(estimators=models)
result=cross_val_score(ensemble_model,X,y,cv=kfold)
print('投票算法(Voting)：',result.mean())










