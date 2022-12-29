import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Data preprocessing!!! >>>>>>>>>>>

data = pd.read_csv('marketing_campaign.csv', delimiter=';')
del data['Dt_Customer']

# One HOT ENCODING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

one_hot_encoder = OneHotEncoder()
# for EDUCATION
feature_array = one_hot_encoder.fit_transform(data[["Education"]]).toarray()
feature_labels = one_hot_encoder.categories_
feature_labels = np.array(feature_labels).ravel()
encoded_education = pd.DataFrame(feature_array, columns=feature_labels)
education_count = len(data["Education"].unique())
del data['Education']
print(data)

for x in range(education_count):
    data.insert(2 + x, feature_labels[x], encoded_education[encoded_education.columns[x]])

# for MARITAL STATUS
feature_array = one_hot_encoder.fit_transform(data[["Marital_Status"]]).toarray()
feature_labels = one_hot_encoder.categories_
feature_labels = np.array(feature_labels).ravel()
encoded_marital = pd.DataFrame(feature_array, columns=feature_labels)
marital_count = len(data["Marital_Status"].unique())
del data['Marital_Status']
print(data)

for x in range(marital_count):
    data.insert(2 + education_count + x, feature_labels[x], encoded_marital[encoded_marital.columns[x]])
# ONE HOT ENCODING END<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

print(data)
# End of data preprocessing<<<<<<<<<<<<<

# Decision Tree Part


# Training and Testing set preperation
y = data['Response']
del data['Response']
X = data.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
y_train.fillna(y_train.mean(), inplace=True)
y_test.fillna(y_test.mean(), inplace=True)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('----------------------------------------------------------------------------------------')
predictions = clf.predict(X_test)

print("Accuracy of classification:")
print(accuracy_score(y_test, predictions))


'''
# Visualisation of classification
fig, ax = plt.subplots(figsize=(70, 70))
tree.plot_tree(clf, feature_names=X.columns, class_names="Response", filled=True)
plt.show()
'''

# Bagging Classifier

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.67,
    bootstrap=True,
    oob_score=True,
    random_state=0
)

bag_model.fit(X_train, y_train)
print("Accuracy of bagging classifier (training set):")
print(bag_model.oob_score_)
print("Accuracy of bagging classifier (testing set - to see if it overfits):")
# if we're not getting a lower value than training set it does not overfits
print(bag_model.score(X_test, y_test))

# Visualisation
plt.figure(figsize=(30, 20))
plot_tree(bag_model.estimators_[0], feature_names=X.columns)
plt.show()
