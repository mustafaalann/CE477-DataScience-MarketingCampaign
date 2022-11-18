import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

missingValues = ["n/a", "na", "--", "NA", "N/A", " "]
df = pd.read_csv('marketing_campaign.csv', delimiter=";", na_values=missingValues)

def _Correlation():
    data = pd.DataFrame(df)
    print(data.corr())
    plt.figure(figsize=(19, 8))
    sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    plt.show()

_Correlation()