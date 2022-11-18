import matplotlib.pyplot as plt
import pandas as pd

missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)


def boxPlot(x):  # boxplot include minimum, first quartile (Q1), median, third quartile (Q3), and maximum
    df[x].dropna().plot(kind='box', title=x)
    plt.show()

boxPlot("MntWines")
boxPlot("Recency")
boxPlot("Income")
boxPlot("Teenhome")
