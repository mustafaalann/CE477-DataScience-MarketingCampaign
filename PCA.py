
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot
from sklearn.preprocessing import Normalizer

df = pd.read_csv("marketing_campaign.csv", delimiter=';')
columnsToRescale = df[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                      'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                      'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']].dropna()
print('Before Normalization')
print(columnsToRescale)

#Normalization
scaler = Normalizer().fit(columnsToRescale)
normalizedData = scaler.transform(columnsToRescale)
normalizedData = pd.df(normalizedData,index=columnsToRescale.index,columns=columnsToRescale.columns)
print('After Normalization')
print(normalizedData)

# Reformating and viewing results
loadings = pd.df(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_normalizedData.columns))],
index=df.columns)
print(loadings)

plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()
