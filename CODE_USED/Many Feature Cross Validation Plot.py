# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:11:24 2021

@author: oran

@brief: code for plotting cross-validations of many-feature regression models
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
    if filename == "AllFeatures_2011-2015":
        X1 = df.iloc[:,0]
        X2 = df.iloc[:,1]
        X3 = df.iloc[:,2]
        X4 = df.iloc[:,3]
        X5 = df.iloc[:,4]
        X6 = df.iloc[:,5]
        X7 = df.iloc[:,6]
        X8 = df.iloc[:,7]
        X9 = df.iloc[:,8]
        X10 = df.iloc[:,9]
        X11 = df.iloc[:,10]
        
        X_val = np.column_stack((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11))
        
        y_val = df.iloc[:,11]  
    elif filename == "SomeFeatures_2006-2016":
        X1 = df.iloc[:,0]
        X2 = df.iloc[:,1]
        X3 = df.iloc[:,2]
        X4 = df.iloc[:,3]
        X5 = df.iloc[:,4]
        X6 = df.iloc[:,5]
        X7 = df.iloc[:,6]
        X8 = df.iloc[:,7]
        X9 = df.iloc[:,8]
        
        X_val = np.column_stack((X1,X2,X3,X4,X5,X6,X7,X8,X9))
        
        y_val = df.iloc[:,9]  
        
       
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
    plt.title("Ridge Regression, q = "+str(q1)+", Applied to\n"+filename)
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
    plt.title("Lasso Regression, q = "+str(q1)+", Applied to\n"+filename)
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
    plt.title("Random Forest Regression, q = "+str(q3)+", Applied to\n"+filename)
    plt.savefig("Cross Val/Plots/"+filename+" Random Forest.pdf")
    plt.show()
      
    
if os.path.isfile("Cross Val/Plots/Boole.txt") == False:
    os.mkdir("Cross Val/Plots")

    f = open("Cross Val/Plots/Boole.txt",'w')
    f.close()

CrossValPlot("AllFeatures_2011-2015",3,1,3)
CrossValPlot("SomeFeatures_2006-2016",2,3,1)