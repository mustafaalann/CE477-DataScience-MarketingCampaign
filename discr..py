import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer


df = pd.read_csv("marketing_campaign.csv", delimiter=';')
numerics = df[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth']].dropna()
numerics[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth']].hist(bins=15,figsize=(8,6))
print(numerics.shape)
data = numerics.values[:, :-1]
trans = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
data = trans.fit_transform(data)
numerics = DataFrame(data)
print(numerics.describe())
numerics.hist(bins=15,figsize=(8,6))
plt.show()
