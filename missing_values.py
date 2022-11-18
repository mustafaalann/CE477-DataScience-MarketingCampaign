import pandas as pd

missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)

print(df.isnull().sum())