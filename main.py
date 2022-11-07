import pandas as pd

df = pd.read_csv('marketing_campaign.csv')
data=df.parsel("Customer's id")
print(data.head(10))
