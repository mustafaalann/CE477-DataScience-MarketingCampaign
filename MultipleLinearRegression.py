import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime as dt

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

x = df[['Year_Birth','Graduation','PhD','Master','Basic','2n Cycle','Absurd','Alone','Divorced','Married','Single','Together','Widow','YOLO','Income','Kidhome', 'Teenhome','Dt_Customer','Recency','MntWines','MntFruits','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response']]
y = df['MntMeatProducts']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x_train = sm.add_constant(x_train)  # adding a constant

model = sm.OLS(y_train, x_train).fit()
predictions = model.predict(x_test)

print(model.summary())



