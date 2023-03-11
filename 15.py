# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:59:11 2023

@author: hp
"""
from sklearn import linear_model
X=[[20,3],
   [23,7],
   [31,10],
   [42,13],
   [50,7],
   [60,5]]

y=[0,
   1,
   1,
   1,
   0,
   0]

lr=linear_model.LogisticRegression()
lr.fit(X,y)

testX=[[28,8]]

label=lr.predict(testX)
print("predicted Label=",label)

prob=lr.predict_proba(testX)
print("probalility=",prob)
