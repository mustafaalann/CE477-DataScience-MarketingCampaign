import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing as per

#loading data
dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
columnsToRescale = dataset[['Income','MntWines']].dropna()
print('Before Standardization')
print(columnsToRescale)

#Standardization
scaler = per.StandardScaler().fit(columnsToRescale)
standardizedData = scaler.transform(columnsToRescale)
standardizedData = pd.DataFrame(standardizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After Standardization')
print(standardizedData)