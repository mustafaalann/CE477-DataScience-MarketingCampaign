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
X.reset_index(drop=True)



from scipy import stats
X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]




clusterer = DBSCAN(eps=0.3, min_samples=100, metric='euclidean')
y_pred = clusterer.fit_predict(X)

plt.figure(figsize=(12,9))
plt.annotate('Income and MntWines DBSCAN', xy=(0.03, 0.95), xycoords='axes fraction')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='rainbow')
plt.show()
print(len(y_pred))