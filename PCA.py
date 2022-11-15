import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('marketing_campaign.csv')
columns_names = df.columns.tolist()
print("Columns names:")
print(columns_names)
df.shape()
df.head()
df.corr()
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

plt.title('Correlation between different features')
