import pandas as pd
from sklearn.model_selection import train_test_split

# reading dataset from csv file
df = pd.read_csv("marketing_campaign.csv", delimiter=';')

# Selecting columns
x = df.iloc[:, :-1] #until the last column of data frame
y = df.iloc[:, -1] #the last column of data frame(Z_Revenue)

# splitting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.80, random_state=0)
print(x_train)
print(x_test)
