#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:41:53 2021

@author: oran

@brief: a script to check the correlation of single features against the target
data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

target_name = "Crime_2000-2017"

def Correlation(filename):
    feature = pd.read_csv("Data/"+filename+".csv")
    target = pd.read_csv("Data/"+target_name+".csv")
    
    target_cols = target.columns
    target_years = []
    
    for i in target_cols:
        target_years.append(i)
    
    feature_cols = feature.columns
    feature_years = []
    
    for i in feature_cols:
        feature_years.append(i)
    
    if min(target_years) < min(feature_years):
        begin = min(feature_years)
    else:
        begin = min(target_years)
        
    if max(target_years) > max(feature_years):
        end = max(feature_years)
    else:
        end = max(target_years)
    
    begin = int(float(begin))
    end = int(float(end))
    
    temp = np.linspace(begin,end,end-begin+1)
    years = []
    
    for i in temp:
        years.append(str(int(i)))
    
    report = filename.split('_')
    f = open("Correlation Reports/"+report[0]+"_Report.txt",'w')
    
    boole = 0
    
    for i in years:
        feat_temp = np.array(feature[i])
        tar_temp = np.array(target[i])
        feat = []
        tar = []
        
        for j in range(len(feat_temp)):
            if feat_temp[j] != '#':
                feat.append(float(feat_temp[j]))
                tar.append(tar_temp[j])
            else:
                if boole == 0:
                    boole = 1
        
        plt.scatter(feat,tar,s=7,label=i)
        plt.xlabel(report[0])
        plt.ylabel("Crimerate")
        if boole == 0:
            correlate = feature[i].corr(target[i])
        else:
            correlate = np.corrcoef(feat,tar)[0,1]
        f.write("Correlation of metric to borough in year "+i+": "+"%.3f"%correlate+"\n")
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig("Correlation Plots/Crime_vs_"+filename+".pdf")
    plt.show()
    
    f.close()

Correlation("ChildPov<16_2006-2016")
Correlation("ChildPov<18_2006-2016")
Correlation("Female65LifeExpect_2000-2017")
Correlation("FemaleLifeExpect_2000-2017")
Correlation("Male65LifeExpect_2000-2017")
Correlation("MaleLifeExpect_2000-2017")
Correlation("NEETs_2009-2015")
Correlation("Rents_2011-2017")
Correlation("Traffic_2010-2017")
Correlation("Workless_2004-2017")
Correlation("Earnings_2000-2017")