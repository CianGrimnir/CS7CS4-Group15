# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:11:24 2021

@author: oran

@brief: code for training and testing two-feature regression models
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
    
    X1 = df.iloc[:,0]
    X2 = df.iloc[:,1]
    y_train = df.iloc[:,2]
    
    X_train = np.column_stack((X1,X2))
        
    y_train = np.array(y_train)
    
    df = pd.read_csv("TrainTestVal/"+filename+"_test.csv")
    
    X1 = df.iloc[:,0]
    X2 = df.iloc[:,1]
    y_test = df.iloc[:,2]
    
    X_test = np.column_stack((X1,X2))
        
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

Model("ChildPov16&Rents_2011-2016",[1,10],[2,1000],[2,1000])
Model("Workless&Traffic_2004-2017",[1,1],[1,1000],[3,100])