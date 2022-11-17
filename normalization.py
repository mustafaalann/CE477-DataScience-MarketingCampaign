import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing as per

#loading data
from sklearn.preprocessing import Normalizer

dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
columnsToRescale = dataset[['Income','MntWines']].dropna()
print('Before Normalization')
print(columnsToRescale)

#Normalization
scaler = Normalizer().fit(columnsToRescale)
normalizedData = scaler.transform(columnsToRescale)
normalizedData = pd.DataFrame(normalizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After Normalization')
print(normalizedData)