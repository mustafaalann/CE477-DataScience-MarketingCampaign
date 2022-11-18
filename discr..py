import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

df = pd.read_csv("marketing_campaign.csv",delimiter=';')
print(df.shape)

print(df.describe())

df.hist()
pyplot.show()
