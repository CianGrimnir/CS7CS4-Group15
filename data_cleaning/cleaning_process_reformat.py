#!/usr/bin/env python

import pandas as pd

df = pd.read_csv('life_expectancy_2002_18.csv')

male = df.iloc[:,3]
female = df.iloc[:,4]
male_65 = df.iloc[:,5]
female_65 = df.iloc[:,6]

index = list(range(0,32))


def write_data(filename, dataset, index):
	f=open(filename,'w')
    index=32
    j=0
	for i in range(len(dataset)):
        if j==index:
            f.write("\n")
            j=0
		f.write(str(dataset[j])+"\t")
		j+=1
	f.close()


male_filename="cleaned_data_for_male_life_expectancy_2002_18.csv"
female_filename="cleaned_data_for_female_life_expectancy_2002_18.csv"
male_at_65_filename="cleaned_data_for_male_at_65_life_expectancy_2002_18.csv"
female_at_65_filename="cleaned_data_for_female_at_65_life_expectancy_2002_18.csv"

write_data(male_filename,male,index)
write_data(female_filename,female,index)
write_data(male_at_65_filename,male_65,index)
write_data(female_at_65_filename,female_65,index)

