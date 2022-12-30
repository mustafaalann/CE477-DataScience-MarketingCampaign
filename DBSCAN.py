import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing as per
from sklearn.cluster import DBSCAN


#loading data
dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
columnsToRescale = dataset[['Income','MntWines']].dropna()
print('Before Standardization')
print(columnsToRescale)

#Standardization
scaler = per.StandardScaler().fit(columnsToRescale)
standardizedData = scaler.transform(columnsToRescale)
standardizedData = pd.DataFrame(standardizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After StandardizationN')
print(standardizedData)
X = standardizedData

def IQR(x):  # Q3 - Q1
    _IQR = np.quantile(dataset[x].dropna(), 0.75) - np.quantile(dataset[x].dropna(), 0.25)
    return _IQR

def lowerFence(x):  # Q1 - 1.5 * IQR
    lowerFen = np.quantile(dataset[x].dropna(), 0.25) - 1.5 * IQR(x)
    return lowerFen


def upperFence(x):  # Q3 + 1.5 * IQR
    upperFen = np.quantile(dataset[x].dropna(), 0.75) + 1.5 * IQR(x)
    return upperFen

print(X['Income'][0] )
for i in range(len(X)):
    if(X['Income'][i] < lowerFence("Income") or X['Income'][i] > upperFence("Income")):
        X.drop(X['Income'][i])

for i in range(len(X)):
    if(X['MntWines'][i] < lowerFence("MntWines") or X['MntWines'][i] > upperFence("MntWines")):
        X.drop(X['MntWines'][i])






clusterer = DBSCAN(eps=0.3, min_samples=10, metric='euclidean')
y_pred = clusterer.fit_predict(X)

plt.figure(figsize=(12,9))
plt.annotate('CE 477', xy=(0.03, 0.95), xycoords='axes fraction')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='Dark2')
plt.show()