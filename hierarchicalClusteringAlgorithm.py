import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform, pdist
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing as per

#loading data
dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
columnsToRescale = dataset[['MntFruits','MntWines']].dropna()
print('Before Standardization')
print(columnsToRescale)

#Standardization
scaler = per.StandardScaler().fit(columnsToRescale)
standardizedData = scaler.transform(columnsToRescale)
standardizedData = pd.DataFrame(standardizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After Standardization')
print(standardizedData)


plt.figure(figsize=(8,5))
plt.scatter(standardizedData['MntFruits'], standardizedData['MntWines'], c='r', marker='*')
plt.xlabel('Column MntFruits')
plt.ylabel('column MntWines')
plt.title('Scatter Plot of Income and MntWines')
for j in standardizedData.itertuples():
    plt.annotate(j.Index, (j.MntFruits, j.MntWines), fontsize=0)
plt.show()


dist = pd.DataFrame(squareform(pdist(standardizedData[['MntFruits','MntWines']]), 'euclidean'), columns=standardizedData.index.values, index=standardizedData.index.values)

plt.figure(figsize=(12,5))
plt.title("Dendrogram with Single")
dend = shc.dendrogram(shc.linkage(standardizedData[['MntFruits','MntWines']], method='single'), labels=standardizedData.index)
plt.show()