
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('marketing_campaign.csv',delimiter=';')
numerics = dataset[["Income","Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts",
             "MntGoldProds","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases",
            "NumWebVisitsMonth"]].dropna()

colours = []
for i in range(2216):
    colours.append(i+1)

scaler = StandardScaler()
scaler.fit(numerics)
scaled_data = scaler.transform(numerics)

print(scaled_data)

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=colours)
plt.xlabel("First Principle Component")
plt.ylabel("Second Principle Component")

plt.show()

