import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)

data = pd.read_csv('marketing_campaign.csv', delimiter=';')
print(data)

training_data = df.sample(frac=0.8, random_state=25)
testing_data = df.drop(training_data.index)

print(f"Number of training examples: {training_data.shape[0]}")
print(f"Number of testing examples: {testing_data.shape[0]}")







# print(data.Income) for see the just Income attribute

# print(df.to_string())
# print(df.isnull().sum()) this is for number of missing values
'''
df['Income'] = df['Income'].fillna(df['Income'].mean())
print(df.isnull().sum()) # this is for number of missing values after imputing with mean
'''

def _Correlation():
    data = pd.DataFrame(df)
    print(data.corr())
    plt.figure(figsize=(19, 8))
    sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    plt.show()

#_Correlation()



