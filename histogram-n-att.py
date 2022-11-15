import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("marketing_campaign.csv",delimiter=';')
print(df.head(2240))
print(df.columns)
df[['Education']].value_counts().plot(kind='bar',figsize=(8,6),fontsize=(10),alpha=0.4)
plt.xticks(rotation='horizontal')
plt.title('Education')
plt.xlabel('Education')
plt.ylabel('Frequency')
plt.show()

df[['Marital_Status']].value_counts().plot(kind='bar',figsize=(8,6),fontsize=(10),alpha=0.4)
plt.title('Marital Status')
plt.xticks(rotation='horizontal')
plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.show()
