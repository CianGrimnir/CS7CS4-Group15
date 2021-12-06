# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:11:24 2021

@author: oran

@brief: code for training and testing one-feature regression models
"""

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

kf = sk.model_selection.KFold(n_splits=5)

def Model(filename,hypers1,hypers2,hypers3):
    #parse dataset
    df = pd.read_csv("TrainTestVal/"+filename+"_train.csv")
    
    feature = df.iloc[:,0]
    y_train = df.iloc[:,1]
    
    X_train = []
    for i in feature:
        X_train.append([i])
        
    y_train = np.array(y_train)
    
    df = pd.read_csv("TrainTestVal/"+filename+"_test.csv")
    
    feature = df.iloc[:,0]
    y_test = df.iloc[:,1]
    
    X_test = []
    for i in feature:
        X_test.append([i])
        
    y_test = np.array(y_test)
    
    f = open("Model Reports/"+filename+".txt",'w')
    
    #Ridge
    f.write("================\nRidge Regression\n================\n")
    
    q = hypers1[0]
    C = hypers1[1]
    
    poly_X_train = sk.preprocessing.PolynomialFeatures(q).fit_transform(X_train)
    poly_X_test = sk.preprocessing.PolynomialFeatures(q).fit_transform(X_test)
    
    model = Ridge(alpha=1/(2*C)) 
    model.fit(poly_X_train,y_train)

    score = model.score(poly_X_test,y_test)
    
    f.write("q = "+str(q)+", C = "+str(C)+"\n")
    f.write("Score: "+str(score)+"\n") 

    #Lasso
    f.write("\n================\nLasso Regression\n================\n")
    
    q = hypers2[0]
    C = hypers2[1]
    
    poly_X_train = sk.preprocessing.PolynomialFeatures(q).fit_transform(X_train)
    poly_X_test = sk.preprocessing.PolynomialFeatures(q).fit_transform(X_test)
    
    model = Lasso(alpha=1/(2*C))
    model.fit(poly_X_train,y_train)

    score = model.score(poly_X_test,y_test)
    
    f.write("q = "+str(q)+", C = "+str(C)+"\n")
    f.write("Score: "+str(score)+"\n") 
    
    #RandomForest
    f.write("\n========================\nRandom Forest Regression\n========================\n")
    
    q = hypers3[0]
    estimator = hypers3[1]
    
    poly_X_train = sk.preprocessing.PolynomialFeatures(q).fit_transform(X_train)
    poly_X_test = sk.preprocessing.PolynomialFeatures(q).fit_transform(X_test)
    
    model = RandomForestRegressor(n_estimators=estimator) 
    model.fit(poly_X_train,y_train)

    score = model.score(poly_X_test,y_test)
    
    f.write("q = "+str(q)+", C = "+str(C)+"\n")
    f.write("Score: "+str(score)+"\n") 
        
    #Dummy
    f.write("\n================\nDummy Regression\n================\n")
    
    model = sk.dummy.DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)

    score = model.score(X_test,y_test)

    f.write("Score: "+str(score)) 
      
    
if os.path.isfile("Model Reports/Boole.txt") == False:
    os.mkdir("Model Reports")

    f = open("Model Reports/Boole.txt",'w')
    f.close()

Model("ChildPov16_2006-2016",[1,10],[1,1000],[1,500])
Model("ChildPov18_2006-2016",[1,10],[1,1000],[3,1000])
Model("Earnings_2000-2017",[3,1000],[3,10000],[1,100])
Model("Female65LifeExpect_2001-2017",[2,1000],[2,10000],[1,500])
Model("FemaleLifeExpect_2000-2017",[1,10],[1,1000],[2,500])
Model("Male65LifeExpect_2001-2017",[3,1000],[3,10000],[1,500])
Model("MaleLifeExpect_2000-2017",[1,10],[1,1000],[3,100])
Model("NEETs_2009-2015",[1,1],[1,1],[1,500])
Model("Rents_2011-2017",[2,1],[2,10000],[2,250])
Model("Traffic_2000-2017",[2,1000],[2,10000],[3,250])
Model("Workless_2004-2017",[1,10],[1,10000],[3,100])