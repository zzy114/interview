# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 15:50:13 2018

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
#采取中位数100000作为分隔线
def label(x):
    if x>=100000:
        return 1
    else:
        return 0
Y=Y.apply(label)

from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#测试集、训练集划分
def split_train_test(X,Y, size=0.2):
    X = DataFrame(X)
    Y = Series(Y)
    folds = int(1/size)
    kfold=StratifiedKFold(Y,n_folds=folds,random_state=1)  
    x_train = DataFrame()  
    y_test = Series()   
    y_train = Series()   
    x_test = DataFrame() 
    for train, test in kfold:   
        x_train = x_train.append(X.iloc[train]) 
        y_train = y_train.append(Y.iloc[train])
        y_test = y_test.append(Y.iloc[test])
        x_test = x_test.append(X.iloc[test])
        break  
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = split_train_test(df1,Y)

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
#基于Tree importance筛选30个特征
gbdt=GradientBoostingClassifier(n_estimators=170,learning_rate=0.1,subsample= 0.7,random_state=1)
gbdt.fit(x_train, y_train)
weights=gbdt.feature_importances_
weights_sort=np.argsort(weights)[::-1]
selected = weights_sort[:30]
x_train1 = x_train.iloc[:,selected]
x_test1 = x_test.iloc[:,selected]

#包裹法，sbs降维
from sklearn.base import clone
from itertools import combinations
class SBS():
    def __init__(self,estimator,k_features,scoring=roc_auc_score,test_size=0.2,random_state=1):
        self.scoring=scoring
        self.estimator=clone(estimator)
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state
        self.a = []
        self.b = []
    def fit(self,x,y):
        x_train,y_train,x_test,y_test = split_train_test(x,y,size=self.test_size)
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
    

#基于GBDT的SBS降维
sbs = SBS(estimator = gbdt, k_features = 1)
sbs.fit(x_train1,y_train)
subset = sbs.subsets()[(30-16)]
x_train2 = sbs.transform(x_train1,subset)
x_test2 = sbs.transform(x_test1,subset)





#模型训练
#Logistic回归
#网格搜索
param_test0 = [{'penalty':['l1','l2'],'C':list(range(1,10))}]
gs0 = GridSearchCV(estimator = LogisticRegression(random_state= 1),param_grid = param_test0, scoring='roc_auc',iid=False,cv=5,n_jobs=-1)
gs0.fit(x_train2, y_train)
gs0.grid_scores_, gs0.best_params_, gs0.best_score_

log_reg = LogisticRegression(penalty='l2',C=1,random_state= 1)
log_reg.fit(x_train2, y_train)
pred0 = log_reg.predict(x_test2)
#准确率
roc_score0 = roc_auc_score(y_test, pred0)   
acc_score0 = accuracy_score(y_test, pred0) 



#GBDT
#网格搜索
param_test1 = [{'n_estimators':list(range(10,201,10)),'subsample':[x/10 for x in range(1,10)]}]
gs1 = GridSearchCV(estimator = GradientBoostingClassifier(),param_grid = param_test1, n_jobs=-1,scoring='roc_auc',iid=False,cv=5)
gs1.fit(x_train2, y_train)
gs1.grid_scores_, gs1.best_params_, gs1.best_score_  

gbdt=GradientBoostingClassifier(n_estimators=60,learning_rate=0.1,subsample= 0.7,random_state=1)
gbdt.fit(x_train2, y_train)
pred1 = gbdt.predict(x_test2)
#准确率
roc_score1 = roc_auc_score(y_test, pred1)  
acc_score1 = accuracy_score(y_test, pred1)    

#SVM
#网格搜索'poly','linear', 'rbf', 'sigmoid', 'precomputed'
param_test2 = [{'kernel':['linear', 'rbf', 'sigmoid'],'C':list(range(1,10))}]
gs2 = GridSearchCV(estimator = SVC(random_state=1),param_grid = param_test2, scoring='roc_auc',iid=False,cv=5)
gs2.fit(x_train2, y_train)
gs2.grid_scores_, gs2.best_params_, gs2.best_score_  

svm = SVC(kernel='linear', degree=3, coef0=2, C=4)
svm.fit(x_train2, y_train)
pred2 = svm.predict(x_test2)
#准确率
roc_score2 = roc_auc_score(y_test, pred2)   
acc_score42 = accuracy_score(y_test, pred2)
