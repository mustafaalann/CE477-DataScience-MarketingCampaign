import pandas as pd
<<<<<<< Updated upstream

data = pd.read_csv('marketing_campaign.csv', delimiter=';')

print(data)
df = pd.read_csv('marketing_campaign.csv')
=======
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)


# print("Q3 quantile of array : ", np.quantile(df["MntWines"], .75))  DENEME

def IQR(x):  # Q3 - Q1
    _IQR = np.quantile(df[x].dropna(), 0.75) - np.quantile(df[x].dropna(), 0.25)
    return _IQR


def lowerFence(x):  # Q1 - 1.5 * IQR
    lowerFen = np.quantile(df[x], 0.25) - 1.5 * IQR(x)
    return lowerFen


def upperFence(x):  # Q3 + 1.5 * IQR
    upperFen = np.quantile(df[x], 0.75) + 1.5 * IQR(x)
    return upperFen


def boxPlot(x):  # boxplot include minimum, first quartile (Q1), median, third quartile (Q3), and maximum
    df[x].dropna().plot(kind='box', title=x)
    plt.show()


def _Correlation():
    data = pd.DataFrame(df)
    print(data.corr())
    plt.figure(figsize=(19, 8))
    sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    plt.show()


# print(df.to_string())
# print(df.isnull().sum())

# print(IQR("MntWines"))
print(lowerFence("MntWines"))
print(upperFence("MntWines"))

# boxPlot("MntWines")
# boxPlot("Recency")
# boxPlot("Income")

#_Correlation()

# outlier için bir attribute yeterli.


plt.hist(df["Year_Birth"])
plt.title("Birth Year")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(df["Education"])
plt.title("Education")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
>>>>>>> Stashed changes
