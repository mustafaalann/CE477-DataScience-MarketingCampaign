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


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

x = data['MntMeatProducts'].values.reshape(-1,1)
y = data['NumCatalogPurchases'].values.reshape(-1,1)

#Creation of the line
linear_reg.fit(x,y)

p0 = linear_reg.intercept_
p1 = linear_reg.coef_

#Predictions
y_head = linear_reg.predict(x)

print("Prediction P0 :",p0)
print("Prediction P1 :",p1)

#Prediction score
from sklearn.metrics import r2_score
print("R Square Values :",r2_score(y,y_head))

plt.scatter(x, y, color = 'red')
plt.plot(x, y_head, color = 'blue')
plt.title('Linear Regression Algorithm Model')
plt.xlabel('MntMeatProducts')
plt.ylabel('NumCatalogPurchases')
plt.show()

