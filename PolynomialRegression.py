import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime as dt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)

training_data = df.sample(frac=0.8, random_state=25)
testing_data = df.drop(training_data.index)

print(f"Number of training examples: {training_data.shape[0]}")
print(f"Number of testing examples: {testing_data.shape[0]}")

# ONE HOT ENCODING PARTS TO INCLUDE THEM IN REGRESSION

one_hot_encoder = OneHotEncoder()

feature_array_Education = one_hot_encoder.fit_transform(df[["Education"]]).toarray()
feature_labels_Education = one_hot_encoder.categories_
feature_labels_Education = np.array(feature_labels_Education).ravel()
final_array_Education = pd.DataFrame(feature_array_Education,columns=feature_labels_Education)

feature_array_MaritalStatus = one_hot_encoder.fit_transform(df[["Marital_Status"]]).toarray()
feature_labels_MaritalStatus = one_hot_encoder.categories_
feature_labels_MaritalStatus = np.array(feature_labels_MaritalStatus).ravel()
final_array_MaritalStatus = pd.DataFrame(feature_array_MaritalStatus,columns=feature_labels_MaritalStatus)

#filled income with mean
df['Income'] = df['Income'].fillna(df['Income'].mean())

#added one hot encoded education and marital status
df = df.join(final_array_MaritalStatus)
df = df.join(final_array_Education)

#Turning date into numerical value
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Dt_Customer']=df['Dt_Customer'].map(dt.datetime.toordinal)

# 'Year_Birth','Graduation','PhD','Master','Basic','2n Cycle','Absurd','Alone','Divorced','Married','Single','Together','Widow','YOLO','Income','Kidhome', 'Teenhome','Dt_Customer','Recency','MntWines','MntFruits','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response'

x = df[['Year_Birth','Graduation','PhD','Master','Basic','2n Cycle','Absurd','Alone','Divorced','Married','Single','Together','Widow','YOLO','Income','Kidhome', 'Teenhome','Dt_Customer','Recency','MntWines','MntFruits','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response']].values
y = df['MntMeatProducts'].values

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# splitting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=0)

#define our polynomial model, with whatever degree we want
degree=2

# PolynomialFeatures will create a new matrix consisting of all polynomial combinations
# of the features with a degree less than or equal to the degree we just gave the model (2)
poly_model = PolynomialFeatures(degree=degree)

# transform out polynomial features
poly_x_values = poly_model.fit_transform(x)                                                # x train verince işe yaramadı

# should be in the form [1, a, b, a^2, ab, b^2]
print(f'initial values {x[0]}\nMapped to {poly_x_values[0]}')

# [1, a=5, b=2940, a^2=25, 5*2940=14700, b^2=8643600]
# let's fit the model
poly_model.fit(poly_x_values, y)

# we use linear regression as a base!!! ** sometimes misunderstood **
regression_model = LinearRegression()

regression_model.fit(poly_x_values, y)

y_pred = regression_model.predict(poly_x_values)

coef= regression_model.coef_

print(coef)

mean_squared_error(y, y_pred, squared=False)

number_degrees = [1, 2, 3, 4, 5, 6, 7]
plt_mean_squared_error = []
for degree in number_degrees:
    poly_model = PolynomialFeatures(degree=degree)

    poly_x_values = poly_model.fit_transform(x)
    poly_model.fit(poly_x_values, y)

    regression_model = LinearRegression()
    regression_model.fit(poly_x_values, y)
    y_pred = regression_model.predict(poly_x_values)

    plt_mean_squared_error.append(mean_squared_error(y, y_pred, squared=False))

plt.scatter(number_degrees, plt_mean_squared_error, color="green")
plt.plot(number_degrees, plt_mean_squared_error, color="red")
