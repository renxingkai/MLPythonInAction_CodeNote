# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:47:44 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix

filename='diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']

data=pd.read_csv(filename)
#数据分布的直方图
data.hist()
plt.show()

#数据分布的密度图
''''
sharex
sharey
分别代表是否共享x,y轴
'''
data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
plt.show()


#数据分布的箱线图
data.plot(kind='box',subplots=True,layout=(3,3),sharex=False)
plt.show()

'''
做出相关矩阵图
'''
#获取相关系数矩阵
correlations=data.corr()
fig=plt.figure()
ax=fig.add_subplot(111)

#matshow用来将矩阵可视化
cax=ax.matshow(correlations,vmin=-1,vmax=1)
#colorbar设置渐变色
fig.colorbar(cax)
ticks=np.arange(0,9,1)
#修改x、y的刻度
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#设置x、y轴的标签文字，此处names为列表，取出列表中各个元素
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


'''
绘制散点矩阵图:因变量随自变量变化的大致趋势
'''
scatter_matrix(data)
plt.show()






















