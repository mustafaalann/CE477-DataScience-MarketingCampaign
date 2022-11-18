import pandas as pd
import numpy as np


missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)


def outlierFind(x):
    outliers = []
    for item in df[x]:
        if(item<lowerFence(x) or item>upperFence(x)):
            outliers.append(item)
    outliers.sort()
    return outliers

def IQR(x):  # Q3 - Q1
    _IQR = np.quantile(df[x].dropna(), 0.75) - np.quantile(df[x].dropna(), 0.25)
    return _IQR


def lowerFence(x):  # Q1 - 1.5 * IQR
    lowerFen = np.quantile(df[x].dropna(), 0.25) - 1.5 * IQR(x)
    return lowerFen


def upperFence(x):  # Q3 + 1.5 * IQR
    upperFen = np.quantile(df[x].dropna(), 0.75) + 1.5 * IQR(x)
    return upperFen

def printAll (x) :
    print("IQR of attribute " + x + ": ")
    print(IQR(x))
    print("Lower fence of " + x + ": ")
    print(lowerFence(x))
    print("Upper fence of " + x + ": ")
    print(upperFence(x))
    print("Number of outliers: ")
    print(len(outlierFind(x)))
    print("Outliers: ")
    print(outlierFind(x))
    print()


printAll("MntWines")
printAll("Year_Birth")