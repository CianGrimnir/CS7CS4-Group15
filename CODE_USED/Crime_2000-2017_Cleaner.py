# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:10:12 2021

@author: Oran

@brief: a script for running through the preprocessed Crime_2000-2017_Preprocessed
file to output a .csv file in the desired standard format for the project
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Crime_2000-2017_Preprocessed.csv")

X = df.iloc[:,4]

Rates = []

for i in X:
    Rates.append(i)

f = open("Crime_2000-2017_Clean.csv", 'w')

index = list(range(0,31))

years = np.linspace(2000,2017,18)

for i in years:
    if i == 2000:
        f.write(str(int(i)))
    else:
        f.write(","+str(int(i)))

f.write("\n")

for i in index:
    j = i
    
    while(j<len(Rates)):
        if j == i:
            f.write(str(Rates[j]))
        else:
            f.write(","+str(Rates[j]))
        j += 31
    f.write("\n")
    
f.close()