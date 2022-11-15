import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('marketing_campaign.csv',delimiter=';')

#Here we add the numeric attributes into a list.(We could also get them from data as numeric columns)
numeric_attributes = ["Income","Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts",
             "MntGoldProds","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases","NumStorePurchases",
            "NumWebVisitsMonth"]

for i in range(len(numeric_attributes)):
    for j in range(len(numeric_attributes)):
        if i != j:
            first_attribute = data[numeric_attributes[i]]
            second_attribute = data[numeric_attributes[j]]
            plt.scatter(x=numeric_attributes[i], y=numeric_attributes[j], data=data, edgecolors='black')
            plt.xlabel(numeric_attributes[i])
            plt.ylabel(numeric_attributes[j])
            plt.title(numeric_attributes[i] + " AND " + numeric_attributes[j] + " GRAPH")
            plt.show()

