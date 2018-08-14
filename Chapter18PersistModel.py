# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:02:56 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pickle import dump
from pickle import load 

from sklearn.externals import joblib


'''
通过pickle序列化和反序列化机器学习模型
'''

#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]

test_size=0.33
seed=4

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)

#训练模型
model=LogisticRegression()
model.fit(X_train,y_train)

'''
保存模型
'''
model_file='finalized_mode.sav'
#二进制形式写入
with open(model_file,'wb') as model_f:
    #模型序列化
    dump(model,model_f)

'''
加载模型
'''
with open(model_file,'rb') as model_f:
    #模型反序列化
    loaded_model=load(model_f)
    result=loaded_model.score(X_test,y_test)
    print('算法评估结果:%.3f%%'%(result*100))
    
    
    
    
'''
通过joblib序列化和反序列化机器学习模型
joblib序列化时会采用Numpy的格式保存数据
'''
test_size=0.33
seed=4

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)

#训练模型
model=LogisticRegression()
model.fit(X_train,y_train)

'''
保存模型
'''
model_file='finalized_model_joblib.sav'
with open(model_file,'wb') as model_f:
    joblib.dump(model,model_f)

'''
加载模型
'''
with open(model_file,'rb') as model_f:
    loaded_model=joblib.load(model_f)
    result=loaded_model.score(X_test,y_test)
    print('算法评估结果:%.3f%%'%(result*100))

