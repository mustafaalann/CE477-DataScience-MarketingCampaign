import pandas as pd
from sklearn.model_selection import train_test_split

# reading dataset from csv file
df = pd.read_csv("marketing_campaign.csv", delimiter=';')

# getting the locations
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
print(X_train)
print(X_test)
