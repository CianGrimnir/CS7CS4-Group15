# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:20:49 2021

@author: Oran

@brief: a script for running through the preprocessed ChildPov_2006-2017_Preprocessed
file to output a .csv file in the desired standard format for the project
"""

import pandas as pd
import numpy as np

df = pd.read_csv("ChildPov_2006-2016_Preprocessed.csv")

X16 = df.iloc[:,7]
X18 = df.iloc[:,8]

Rates16 = []

for i in X16:
    Rates16.append(float(i.rstrip("%"))/100)

Rates18 = []

for i in X18:
    Rates18.append(float(i.rstrip("%"))/100)

f16 = open("ChildPov<16_2006-2017_Clean.csv", 'w')
f18 = open("ChildPov<18_2006-2017_Clean.csv", 'w')

index = list(range(0,31))

years = np.linspace(2006,2016,11)

for i in years:
    if i == 2006:
        f16.write(str(int(i)))
        f18.write(str(int(i)))
    else:
        f16.write(","+str(int(i)))
        f18.write(","+str(int(i)))

f16.write("\n")
f18.write("\n")

for i in index:
    j = i
    
    while(j<len(Rates16)):
        if j == i:
            f16.write("%.3f" %Rates16[j])
            f18.write("%.3f" %Rates18[j])
        else:
            f16.write(","+"%.3f" %Rates16[j])
            f18.write(","+"%.3f" %Rates18[j])
        j += 31
    f16.write("\n")
    f18.write("\n")
    
f16.close()
f18.close()