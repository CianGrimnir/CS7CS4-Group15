# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:30:34 2021

@author: Oran

@brief: a script for running through the preprocessed Rents_2011-2017_Preprocessed
file to output a .csv file in the desired standard format for the project
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Rents_2011-2017_Preprocessed.csv')
quarters = df.iloc[3,:]
store_column = []
for i in range(len(quarters)):
    if(quarters[i] == "Q3"):
        store_column.append(i)

f = open("Rents_2011-2017_Clean.csv", 'w')

years = np.linspace(2011,2018,8)

for i in years:
    if i == 2011:
        f.write(str(int(i)))
    else:
        f.write(","+str(int(i)))

f.write("\n")

for i in range(5,36):

    for j in store_column:
        z = df.iloc[i][j]
        z = z.split(',')
        z = ''.join(z)
        f.write(str(z))
        if j != max(store_column):
            f.write(",")
    f.write("\n")
 
f.close()

