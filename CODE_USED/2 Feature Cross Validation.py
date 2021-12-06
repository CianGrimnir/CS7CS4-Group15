# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:11:24 2021

@author: oran

@brief: code for splitting up datasets in train test and validation sets and 
cross-validating various two-feature regression models.
"""

import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb
import os

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

target_name = "Crime_2000-2017"
kf = sk.model_selection.KFold(n_splits=5)

def GetYears(target_cols, feature_cols):
    target_years = []
    
    for i in target_cols:
        target_years.append(i)
        
    max_feat_years = '2030'
    min_feat_years = '1900'
    
    for i in feature_cols:
        if min(i) > min_feat_years:
            min_feat_years = min(i)
        if max(i) < max_feat_years:
            max_feat_years = max(i)
    
    if min(target_years) < min_feat_years:
        begin = min_feat_years
    else:
        begin = min(target_years)
        
    if max(target_years) > max_feat_years:
        end = max_feat_years
    else:
        end = max(target_years)
    
    begin = int(float(begin))
    end = int(float(end))
    
    temp = np.linspace(begin,end,end-begin+1)
    years = []
    
    for i in temp:
        years.append(str(int(i)))
        
    return years

def PrintSet(filename, which, feat_array, tar_array):
    f = open("TrainTestVal/"+filename+which+".csv",'w')
    f.write("X1, X2, y\n")
    
    for i in range(len(feat_array)):
        for j in feat_array[i]:
            f.write(str(j))
            f.write(",")
        f.write(str(tar_array[i]))
        f.write("\n")
    
    f.close()


def CrossVal(filename1, filename2):
    #parse dataset
    feature1 = pd.read_csv("Useable Datasets/"+filename1+".csv")
    feature2 = pd.read_csv("Useable Datasets/"+filename2+".csv")
    target = pd.read_csv("Useable Datasets/"+target_name+".csv")
    
    target_cols = target.columns
    feature1_cols = feature1.columns
    feature2_cols = feature2.columns
    
    feature_cols = [feature1_cols, feature2_cols]
    
    years = GetYears(target_cols,feature_cols)
    
    X = []
    y = []
    
    for i in range(len(feature1[years[0]])):
        X1_temp = feature1.iloc[i,:]
        X2_temp = feature2.iloc[i,:]
        y_temp = target.iloc[i,:]
        
        for j in years:
            if X1_temp[j] != '#' and X2_temp[j] != '#':
                X.append([float(X1_temp[j]),float(X2_temp[j])])
                y.append(y_temp[j])
        
    X = np.array(X)
    y = np.array(y)
    
    X_train_test, X_val, y_train_test, y_val = sk.model_selection.train_test_split(X, y, test_size= 0.2, random_state=42)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_train_test, y_train_test, test_size= 0.25, random_state=42)
    
    report1 = filename1.split('_')
    report2 = filename2.split('_')
    
    filename = report1[0]+"&"+report2[0]+"_"+str(min(years))+"-"+str(max(years))
    
    PrintSet(filename, "_train", X_train, y_train)
    PrintSet(filename, "_test", X_test, y_test)
    PrintSet(filename, "_val", X_val, y_val)
    
    reportname = report1[0]+"&"+report2[0]
    
    f = open("Cross Val/Reports/"+reportname+"_RidgeRegression_Report.txt",'w')
    
    #Ridge
    C = [0.0001,0.001,0.01,0.1,1,10,100,1000]
    q = [1,2,3]
    
    hyper_grid = []
    
    for Ci in C:
        for qi in q:
            hyper_grid.append([Ci,qi])
    
    f.write("================\nRidge Regression\n================\n\n")
    
    mean_array = []
    std_array = []
    
    for i in hyper_grid:
        poly_X_val = sk.preprocessing.PolynomialFeatures(i[1]).fit_transform(X_val)
        
        model = Ridge(alpha=1/(2*i[0])) 
        
        kfld = 1
        temp = []
           
        f.write("q = "+str(i[1])+", C = "+str(i[0])+"\n\n")
        for train,test in kf.split(poly_X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
            
            score = model.score(poly_X_val[test],y[test])
            f.write("K-fold "+str(kfld)+" score: "+"%.6f"%score+"\n") 
            
            kfld += 1
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        f.write("\nAverage mean squared error: "+"%.6f"%err_mean+"\nStandard deviation of mean squared error: "+"%.6f"%err_std+"\n\n")
        mean_array.append(err_mean)
        std_array.append(err_std)
    
    min_mean = min(mean_array)
    min_mean_index = mean_array.index(min_mean)
    min_mean_std = std_array[min_mean_index]
    
    hyper_best_mean = hyper_grid[min_mean_index]
    
    min_std = min(std_array)
    min_std_index = std_array.index(min_std)
    min_std_mean = mean_array[min_std_index]
    
    hyper_best_std = hyper_grid[min_std_index]
    
    f.write("\nHyperparameters with minimum MSE: q = "+str(hyper_best_mean[1])+", C = "+str(hyper_best_mean[0])+"\nMSE = "+"%.6f"%min_mean+"\nstdev = "+"%.6f"%min_mean_std+"\n")
    f.write("\nHyperparameters with minimum stdev: q = "+str(hyper_best_std[1])+", C = "+str(hyper_best_std[0])+"\nMSE = "+"%.6f"%min_std_mean+"\nstdev = "+"%.6f"%min_std+"\n")
    
    f.close()
    
    #Lasso
    f = open("Cross Val/Reports/"+reportname+"_LassoRegression_Report.txt",'w')
    
    C = [0.01,0.1,1,10,100,1000,10000]
    q = [1,2,3]
    
    hyper_grid = []
    
    for Ci in C:
        for qi in q:
            hyper_grid.append([Ci,qi])
    
    f.write("================\nLasso Regression\n================\n")
    
    mean_array = []
    std_array = []
    
    for i in hyper_grid:
        poly_X_val = sk.preprocessing.PolynomialFeatures(i[1]).fit_transform(X_val)
        
        model = Lasso(alpha=1/(2*i[0]))
        
        kfld = 1
        temp = []
        
        f.write("q = "+str(i[1])+", C = "+str(i[0])+"\n\n")
        for train,test in kf.split(poly_X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
            
            score = model.score(poly_X_val[test],y_val[test])
            f.write("K-fold "+str(kfld)+" score: "+str(score)+"\n") 
            
            kfld += 1
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        f.write("\nAverage mean squared error: "+"%.6f"%err_mean+"\nStandard deviation of mean squared error: "+"%.6f"%err_std+"\n\n")
        mean_array.append(err_mean)
        std_array.append(err_std)
    
    min_mean = min(mean_array)
    min_mean_index = mean_array.index(min_mean)
    min_mean_std = std_array[min_mean_index]
    
    hyper_best_mean = hyper_grid[min_mean_index]
    
    min_std = min(std_array)
    min_std_index = std_array.index(min_std)
    min_std_mean = mean_array[min_std_index]
    
    hyper_best_std = hyper_grid[min_std_index]
    
    f.write("\nHyperparameters with minimum MSE: q = "+str(hyper_best_mean[1])+", C = "+str(hyper_best_mean[0])+"\nMSE = "+"%.6f"%min_mean+"\nstdev = "+"%.6f"%min_mean_std+"\n")
    f.write("\nHyperparameters with minimum stdev: q = "+str(hyper_best_std[1])+", C = "+str(hyper_best_std[0])+"\nMSE = "+"%.6f"%min_std_mean+"\nstdev = "+"%.6f"%min_std+"\n")
    
    f.close()
    
    #RandomForest
    f = open("Cross Val/Reports/"+reportname+"_RandomForestRegression_Report.txt",'w')
    
    estimators = [100, 250, 500, 750, 1000]
    q = [1,2,3]
    
    hyper_grid = []
    
    for Ei in estimators:
        for qi in q:
            hyper_grid.append([Ei,qi])
    
    f.write("========================\nRandom Forest Regression\n========================\n\n")
    
    mean_array = []
    std_array = []
    
    for i in hyper_grid:
        poly_X_val = sk.preprocessing.PolynomialFeatures(i[1]).fit_transform(X_val)
        
        model = RandomForestRegressor(n_estimators=i[0])
        
        kfld = 0
        temp = []
        
        f.write("q = "+str(i[1])+", Estimators = "+str(i[0])+"\n\n")
        for train,test in kf.split(poly_X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
            
            score = model.score(poly_X_val[test],y_val[test])
            f.write("K-fold "+str(kfld)+" score: "+str(score)+"\n") 
            
            kfld += 1
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        f.write("\nAverage mean squared error: "+"%.6f"%err_mean+"\nStandard deviation of mean squared error: "+"%.6f"%err_std+"\n\n")
        mean_array.append(err_mean)
        std_array.append(err_std)
    
    min_mean = min(mean_array)
    min_mean_index = mean_array.index(min_mean)
    min_mean_std = std_array[min_mean_index]
    
    hyper_best_mean = hyper_grid[min_mean_index]
    
    min_std = min(std_array)
    min_std_index = std_array.index(min_std)
    min_std_mean = mean_array[min_std_index]
    
    hyper_best_std = hyper_grid[min_std_index]
    
    f.write("\nHyperparameters with minimum MSE: q = "+str(hyper_best_mean[1])+", Estimator = "+str(hyper_best_mean[0])+"\nMSE = "+"%.6f"%min_mean+"\nstdev = "+"%.6f"%min_mean_std+"\n")
    f.write("\nHyperparameters with minimum stdev: q = "+str(hyper_best_std[1])+", Estimator = "+str(hyper_best_std[0])+"\nMSE = "+"%.6f"%min_std_mean+"\nstdev = "+"%.6f"%min_std+"\n")
    
    f.close()
    
    
    #XGBoost
    f = open("Cross Val/Reports/"+reportname+"_XGBoost_Report.txt",'w')
    
    q = [1,2,3]
    
    f.write("\n==================\nXGBoost Regression\n==================\n\n")
    
    mean_array = []
    std_array = []
    
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
    
    for qi in q:
        poly_X_val = sk.preprocessing.PolynomialFeatures(qi).fit_transform(X_val)
        
        kfld = 0
        temp = []
        
        for train,test in kf.split(X_val):
            model.fit(poly_X_val[train],y_val[train])
            y_pred = model.predict(poly_X_val[test])
            
            mse = sk.metrics.mean_squared_error(y_val[test],y_pred)
            temp.append(mse)
            
            score = model.score(poly_X_val[test],y_val[test])
            f.write("K-fold "+str(kfld)+" score: "+str(score)+"\n") 
            
            kfld += 1
        
        err_mean = np.array(temp).mean()
        err_std =  np.array(temp).std()
        f.write("\nAverage mean squared error: "+"%.6f"%err_mean+"\nStandard deviation of mean squared error: "+"%.6f"%err_std+"\n\n")
        mean_array.append(err_mean)
        std_array.append(err_std)
    
    min_mean = min(mean_array)
    min_mean_index = mean_array.index(min_mean)
    min_mean_std = std_array[min_mean_index]
    
    q_best_mean = q[min_mean_index]
    
    min_std = min(std_array)
    min_std_index = std_array.index(min_std)
    min_std_mean = mean_array[min_std_index]
    
    q_best_std = q[min_std_index]
    
    f.write("\nHyperparameter with minimum MSE: q = "+str(q_best_mean)+"\nMSE = "+"%.6f"%min_mean+"\nstdev = "+"%.6f"%min_mean_std+"\n")
    f.write("\nHyperparameter with minimum stdev: q = "+str(q_best_std)+"\nMSE = "+"%.6f"%min_std_mean+"\nstdev = "+"%.6f"%min_std+"\n")
    
    f.close()
        
if os.path.isfile("Cross Val/Boole.txt") == False:
    os.mkdir("Cross Val")
    os.mkdir("Cross Val/Reports")

    f = open("Cross Val/Boole.txt",'w')
    f.close()
    
if os.path.isfile("TrainTestVal/Boole.txt") == False:
    os.mkdir("TrainTestVal")

    f = open("TrainTestVal/Boole.txt",'w')
    f.close()

CrossVal("ChildPov16_2006-2016","Rents_2011-2017")
CrossVal("Workless_2004-2017","Traffic_2000-2017")