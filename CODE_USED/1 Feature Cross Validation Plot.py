# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:11:24 2021

@author: oran

@brief: code for plotting cross-validations of one-feature regression models
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

def CrossValPlot(filename,q1,q2,q3):
    #parse dataset
    df = pd.read_csv("TrainTestVal/"+filename+"_val.csv")
    
    feature = df.iloc[:,0]
    y_val = df.iloc[:,1]
    
    X_val = []
    for i in feature:
        X_val.append([i])
        
    y_val = np.array(y_val)
    
    #Ridge
    C = [0.001,0.01,0.1,1,10,100,1000]
    
    mean_error = []
    std_error = []
    
    poly_X_val = sk.preprocessing.PolynomialFeatures(q1).fit_transform(X_val)
    
    for Ci in C:
        model = Ridge(alpha=1/(2*Ci)) 
        
        temp = []
           
        for train,test in kf.split(poly_X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        mean_error.append(err_mean)
        std_error.append(err_std)
    
    plt.errorbar(C,mean_error,yerr=std_error)
    plt.xscale("log")
    plt.xlabel('log(C)')
    plt.ylabel('Mean Squared Error')
    plt.title("Ridge Regression, q = "+str(q1)+", Applied to "+filename)
    plt.savefig("Cross Val/Plots/"+filename+" Ridge Regression.pdf")
    plt.show()
    
    #Lasso
    C=[0.001,0.01,0.1,1,10,100,1000,10000]
    
    mean_error = []
    std_error = []
    
    poly_X_val = sk.preprocessing.PolynomialFeatures(q2).fit_transform(X_val)
    
    for Ci in C:
        model = Lasso(alpha=1/(2*Ci))

        temp=[]
        
        for train,test in kf.split(poly_X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        mean_error.append(err_mean)
        std_error.append(err_std)
    
    plt.errorbar(C,mean_error,yerr=std_error)
    plt.xscale("log")
    plt.xlabel('log(C)')
    plt.ylabel('Mean Squared Error')
    plt.title("Lasso Regression, q = "+str(q1)+", Applied to "+filename)
    plt.savefig("Cross Val/Plots/"+filename+" Lasso.pdf")
    plt.show()
    
    #RandomForest
    estimators=[100, 250, 500, 750, 1000]
    
    mean_error = []
    std_error = []
    
    for Ei in estimators:
        model = RandomForestRegressor(n_estimators=Ei)
        
        temp=[]
        
        poly_X_val = sk.preprocessing.PolynomialFeatures(q2).fit_transform(X_val)

        for train,test in kf.split(poly_X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        mean_error.append(err_mean)
        std_error.append(err_std)
    
    plt.errorbar(estimators,mean_error,yerr=std_error)
    plt.xlabel('Estimator')
    plt.ylabel('Mean Squared Error')
    plt.title("Random Forest Regression, q = "+str(q3)+", Applied to "+filename)
    plt.savefig("Cross Val/Plots/"+filename+" Random Forest.pdf")
    plt.show()
      
    
if os.path.isfile("Cross Val/Plots/Boole.txt") == False:
    os.mkdir("Cross Val/Plots")

    f = open("Cross Val/Plots/Boole.txt",'w')
    f.close()

CrossValPlot("ChildPov16_2006-2016",1,1,1)
CrossValPlot("ChildPov18_2006-2016",1,1,3)
CrossValPlot("Earnings_2000-2017",3,3,1)
CrossValPlot("Female65LifeExpect_2001-2017",2,2,1)
CrossValPlot("FemaleLifeExpect_2000-2017",1,1,2)
CrossValPlot("Male65LifeExpect_2001-2017",3,3,1)
CrossValPlot("MaleLifeExpect_2000-2017",1,1,3)
CrossValPlot("NEETs_2009-2015",1,1,1)
CrossValPlot("Rents_2011-2017",2,1,3)
CrossValPlot("Traffic_2000-2017",2,2,3)
CrossValPlot("Workless_2004-2017",1,1,3)