# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:26:14 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

#读取数据
filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)

'''
使用MinMaxScaler类来调整数据尺度
将数据聚集到0附近，方差为1
'''
#将数据分为输入数据和输出数据
array=data.values
X=array[:,0:8]
y=array[:,8]
#将数据缩放到(0,1)
transformer=MinMaxScaler(feature_range=(0,1))

#数据转换
newX=transformer.fit_transform(X)
#确定浮点数字、数组、和numpy对象的显示形式，此处为小数点后3位
np.set_printoptions(precision=3)
print(newX)

'''
正态化数据
使数据符合正正态分布
中位数为0，方差为1
'''
#此处先fit后transform
transformer=StandardScaler().fit(X)
#数据转换
newX=transformer.transform(X)
#设定数据的打印格式
np.set_printoptions(precision=4)
print(newX)


'''
将数据标准化
将每一行数据距离处理成1
适合处理稀疏数据
'''
#fit准备完数据转化的参数后，保存在transformer中
transformer=Normalizer().fit(X)
#数据转换
newX=transformer.transform(X)
#设置数据的打印格式
np.set_printoptions(precision=5)
print(newX)

'''
二值化数据
大于阈值数据设置为1
小于阈值数据设置为0
'''
#二值化数据阈值threshold设为0.0
transformer=Binarizer(threshold=0.0).fit(X)
#数据转换
newX=transformer.transform(X)
np.set_printoptions(precision=6)
print(newX)








