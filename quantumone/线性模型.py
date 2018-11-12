# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 15:50:12 2018

@author: zhyzhang
"""
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
#预处理数据导入
path=r'I:\\processed data.txt'
df=pd.read_table(path,index_col=0,header=0,engine='python',encoding=None,chunksize=None)
df1=df.drop('投资金额',axis=1)
#标签构建
Y=df.iloc[:,0]


from sklearn.metrics import r2_score 
from sklearn.svm import SVR


#测试集、训练集划分
from sklearn.cross_validation import train_test_split  
x_train,x_test,y_train,y_test = train_test_split(df1,Y,test_size = 0.2,random_state = 0) 

#数据标准化
from sklearn import preprocessing
scl=preprocessing.StandardScaler()
#非编码数据标准化
x1 = x_train.iloc[:,[1,6]]
columns=x1.columns
index=x1.index
x1 = DataFrame(scl.fit_transform(x1),index=index,columns=columns)
x_train=x_train.drop(x_train.columns[[1,6]],axis=1)
x_train=pd.concat([x_train,x1],axis=1)

x2 = x_test.iloc[:,[1,6]]
columns=x2.columns
index=x2.index
x2 = DataFrame(scl.transform(x2),index=index,columns=columns)
x_test=x_test.drop(x_test.columns[[1,6]],axis=1)
x_test=pd.concat([x_test,x2],axis=1)


#特征筛选
#包裹法，sbs降维
from sklearn.base import clone
from itertools import combinations
class SBS():
    def __init__(self,estimator,k_features,scoring=r2_score,test_size=0.2,random_state=1):
        self.scoring=scoring
        self.estimator=clone(estimator)
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state
        self.a = []
        self.b = []
    def fit(self,x,y):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = self.test_size,random_state=self.random_state) 
        x_train,y_train,x_test,y_test = x_train.values,y_train.values,x_test.values,y_test.values
        dim=x_train.shape[1]
        self.indices_=tuple(range(dim))
        self.subsets_=[self.indices_]
        score=self._cal_score(x_train,y_train,x_test,y_test,self.indices_)
        self.scores_=[score]
        while dim >self.k_features:
            scores=[]
            subsets=[]
            for p in combinations(self.indices_,r=dim-1):
                score=self._cal_score(x_train,y_train,x_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            best=np.argmax(scores)
            
            self.a.append(dim-1)
            self.b.append(scores[best])
            
            print(dim-1,'维分数为：',scores[best])
            self.indices_=subsets[best]
            self.subsets_.append(self.indices_)
            dim-=1
            self.scores_.append(scores[best])
        return self 

    def transform(self,x,subset):
        return x.iloc[:,list(subset)]
    
    def subsets(self):
        return self.subsets_
    
    #返回维度和分数，用于作图
    def plt(self):
        return self.a,self.b

    def _cal_score(self,x_train,y_train,x_test,y_test,indices):
        self.estimator.fit(x_train[:,indices],y_train)
        y_pred=self.estimator.predict(x_test[:,indices])
        score=self.scoring(y_test,y_pred)
        return score


#基于gbdt的SBS降维
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=100,subsample=1)
sbs = SBS(estimator = gbdt, k_features = 13)
sbs.fit(x_train,y_train)
subset = sbs.subsets()[(36-18)]

x_train1 = sbs.transform(x_train,subset)
x_test1 = sbs.transform(x_test,subset)

#模型训练
#SVM
svr = SVR(C=0.1,kernel='linear', degree=3)
svr.fit(x_train1,y_train)
pred0=svr.predict(x_test1)
score0 = r2_score(y_test, pred0)

#线性回归
from sklearn.linear_model import LinearRegression
ln=LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
ln.fit(x_train,y_train)
pred1=ln.predict(x_test)
score1 = r2_score(y_test, pred1)

#GBDT
gbdt=GradientBoostingRegressor(loss='ls',learning_rate=0.01,n_estimators=90,subsample=1)
gbdt.fit(x_train1,y_train)
pred2=gbdt.predict(x_test1)
score2 = r2_score(y_test, pred2) 
