#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:47:38 2021

@author: oran

@brief: A script for reading in the cleaned datasets, and then outputting 
a .csv values of their normalized values. The normalization scales them from 
0 for minimum metric measured and 1 for the maximum metric measured
"""

import pandas as pd
import numpy as np

def Normalize(filename):
    df = pd.read_csv(filename+"_Clean.csv")
    f = open("Normalized/"+filename+".csv", "w")
    
    years = df.columns
    
    for i in years:
        if i == min(years):
            f.write(str(i))
        else:
            f.write("," + str(i))
    
    f.write("\n")
    
    mins = []
    for i in df.min():
        if i != '#':
            mins.append(i)
    
    maxs = []
    for i in df.max():
        if i != '#':
            maxs.append(float(i))

    minim = min(mins)
    maxim = max(maxs)
    diff = maxim-minim
    
    for j in range(len(df)):
        for i in range(len(df.iloc[j])):
            if df.iloc[j,i] != '#':
                normal = (float(df.iloc[j,i])-minim)/diff
                f.write("%.3f" %normal)
            else:
                f.write("#")
                
            if i!=len(df.iloc[j])-1:
                f.write(",")
                
        f.write("\n")
                
    
    f.close()

Normalize("ChildPov<16_2006-2016")
Normalize("ChildPov<18_2006-2016")
Normalize("Crime_2000-2017")
Normalize("Earnings_2000-2017")
Normalize("Female65LifeExpect_2000-2017")
Normalize("FemaleLifeExpect_2000-2017")
Normalize("Male65LifeExpect_2000-2017")
Normalize("MaleLifeExpect_2000-2017")
Normalize("NEETs_2009-2015")
Normalize("Rents_2011-2017")
Normalize("Traffic_2010-2017")
Normalize("Workless_2004-2017")