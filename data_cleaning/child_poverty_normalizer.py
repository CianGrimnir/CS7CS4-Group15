#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:47:38 2021

@author: oran
"""

import pandas as pd
import numpy as np

df = pd.read_csv("cleaned children in poverty (UNDER 18) 2006-2016.csv")

f = open("normalized cleaned children in poverty (UNDER 18) 2006-2016.csv", "w")

years = np.linspace(2006, 2016, 11)

for i in years:
    if i == 2006:
        f.write(str(i))
    else:
        f.write("," + str(i))

f.write("\n")

print(max(df.max()))

for j in range(len(df)):
    for i in range(len(df.iloc[j])):
        normal = df.iloc[j,i]/max(df.max())
        f.write("%.3f" %normal)
        if i!=10:
            f.write(",")
            
    f.write("\n")
            

f.close()
