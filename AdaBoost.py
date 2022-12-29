from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import seaborn as sns
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

# ADABOOST Classifier

ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                               n_estimators=100,
                               learning_rate=1)
# Train Adaboost Classifier
ada_model = ada_boost.fit(X_train, y_train)

# Predict the response for test dataset
prediction = ada_model.predict(X_test)

print("Accuracy of AdaBoost Ensemble:")
print(metrics.accuracy_score(y_test, prediction))


# Trying Boxplot with another way of AdaBoost

def get_models():
    _models = dict()
    # explore depths from 1 to 10
    for i in range(1, 11):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        _models[str(i)] = AdaBoostClassifier(base_estimator=base)
    return _models


def evaluate_model(model_, _X, _y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    _scores = cross_val_score(model_, _X, _y, scoring='accuracy', cv=cv, n_jobs=-1)
    return _scores


models = get_models()
results, names = list(), list()

for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X_train, y_train)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.title("Training Set")
pyplot.show()

for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X_test, y_test)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.title("Testing Set")
pyplot.show()
