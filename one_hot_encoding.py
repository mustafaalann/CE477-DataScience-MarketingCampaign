import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('marketing_campaign.csv',delimiter=';')

#This is to check how many different Education types
print(df["Education"].unique())

one_hot_encoder = OneHotEncoder()

#Here we actually get the 1's and 0's as a 2d array
feature_array = one_hot_encoder.fit_transform(df[["Education"]]).toarray()

#Getting the labels
feature_labels = one_hot_encoder.categories_

#Getting the labels into array
feature_labels = np.array(feature_labels).ravel()

#Lets make the full array with labels and values
final_array = pd.DataFrame(feature_array,columns=feature_labels)
print("----------ONE HOT ENCODED(EDUCATION)---------")
print(final_array)
