import pandas as pd
import numpy as np

# poverty
f = open("normalized_children_in_poverty_>16_2006_2016.csv", "w")
df = pd.read_csv("cleaned children in poverty (UNDER 16) 2006-2016.csv")
yrs = np.linspace(2006, 2016, 11)
for i in yrs:
    value = str(i) if i == 2006 else f", {str(i)}"
    f.write(value)

f.write("\n")
for i in range(len(df)):
    for j in range(len(df.iloc[i])):
        normalize = df.iloc[i, j] / max(df.max())
        f.write("%.3f" % normalize)
        if j != 10:
            f.write(",")
    f.write("\n")
f.close()

# life expectancy

df = pd.read_csv("cleaned_data_for_female_at_65_life_expectancy_2002_18.csv")
years = np.linspace(2002, 2018, 17)

f = open("normalized_data_for_female_at_65_life_expectancy_2002_18.csv", "w")

for i in years:
    value = str(i) if i == 2002 else f", {str(i)}"
    f.write(value)
f.write("\n")

for i in range(len(df)):
    for j in range(len(df.iloc[i])):
        normalize = df.iloc[i, j] / max(df.max())
        f.write("%.3f" % normalize)
        if j != 16:
            f.write(",")
    f.write("\n")
f.close()

# average rent

df = pd.read_csv("cleaned-voa-average-rent-borough-Q3-(2011-2018).csv")
years = np.linspace(2011, 2018, 8)
f = open("normalized-voa-average-rent-borough-Q3-(2011-2018).csv", "w")

for i in years:
    value = str(i) if i == 2011 else f", {str(i)}"
    f.write(value)
f.write("\n")

for i in range(len(df)):
    for j in range(len(df.iloc[i])):
        normalize = df.iloc[i, j] / max(df.max())
        f.write("%.3f" % normalize)
        if j != 7:
            f.write(",")
    f.write("\n")
f.close()
