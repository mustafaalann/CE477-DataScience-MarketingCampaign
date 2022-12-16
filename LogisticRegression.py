import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#Data preprocessing!!! >>>>>>>>>>>
missingValues = ["n/a", "na", "--", "NA", "N/A", "NaN", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)

data = pd.read_csv('marketing_campaign.csv', delimiter=';')
del data['Dt_Customer']


#One HOT ENCODING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
one_hot_encoder = OneHotEncoder()
# for EDUCATION
feature_array = one_hot_encoder.fit_transform(data[["Education"]]).toarray()
feature_labels = one_hot_encoder.categories_
feature_labels = np.array(feature_labels).ravel()
encoded_education = pd.DataFrame(feature_array,columns=feature_labels)
education_count = len(data["Education"].unique())
del data['Education']
print(data)

for x in range(education_count):
    data.insert(2+x, feature_labels[x], encoded_education[encoded_education.columns[x]])


# for MARITAL STATUS
feature_array = one_hot_encoder.fit_transform(data[["Marital_Status"]]).toarray()
feature_labels = one_hot_encoder.categories_
feature_labels = np.array(feature_labels).ravel()
encoded_marital = pd.DataFrame(feature_array, columns=feature_labels)
marital_count = len(data["Marital_Status"].unique())
del data['Marital_Status']
print(data)

for x in range(marital_count):
    data.insert(2+education_count+x, feature_labels[x], encoded_marital[encoded_marital.columns[x]])
#ONE HOT ENCODING END<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

print(data)
#End of data preprocessing<<<<<<<<<<<<<

#Decision Tree Part


#Training and Testing set preperation
y = data['Response']
del data['Response']
X = data.copy()
X['Income'] = X['Income'].fillna(X['Income'].mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)
y_test.fillna(y_test.mean(), inplace=True)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, y_train)

print("---------------------------------")
print("Intercept :")
print(model.intercept_)
print("---------------------------------")
print("Coef :")
print(model.coef_)
print("---------------------------------")
print("Score :")
print(model.score(X_test,y_test))

