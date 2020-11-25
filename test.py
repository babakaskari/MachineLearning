<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
sns.set()

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("Acoustic Logger Data.csv")
df1 = df.loc[df["LvlSpr"] == "Lvl"]
df3 = df.loc[df["LvlSpr"] == "Spr"]
df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df5 = pd.merge(df2, df4, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df6 = df5.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df6['Date'] = pd.to_datetime(df6['Date'], format='%d-%b')
df6['Date'] = df6['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df7 = df7.rename(columns={'Date Visited': 'Date'})

df8 = pd.merge(df6, df7, on=['ID', 'Date'], how='left')
df8 = df8.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
dataset["Leak Found"].replace(["Y", "N", "N-PRV"], [1, 0, -2], inplace=True)
dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
print("dataset : ", dataset)
print(dataset.isna().sum())
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

corr = dataset[["value_Lvl", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[0, 0])

corr = dataset[["value_Spr", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[0, 1])

corr = dataset[["Leak Alarm", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[1, 0])

plt.show()
=======
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
sns.set()

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("Acoustic Logger Data.csv")
df1 = df.loc[df["LvlSpr"] == "Lvl"]
df3 = df.loc[df["LvlSpr"] == "Spr"]
df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df5 = pd.merge(df2, df4, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df6 = df5.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df6['Date'] = pd.to_datetime(df6['Date'], format='%d-%b')
df6['Date'] = df6['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df7 = df7.rename(columns={'Date Visited': 'Date'})

df8 = pd.merge(df6, df7, on=['ID', 'Date'], how='left')
df8 = df8.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
dataset["Leak Found"].replace(["Y", "N", "N-PRV"], [1, 0, -2], inplace=True)
dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
print("dataset : ", dataset)
print(dataset.isna().sum())
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

corr = dataset[["value_Lvl", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[0, 0])

corr = dataset[["value_Spr", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[0, 1])

corr = dataset[["Leak Alarm", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[1, 0])

plt.show()
>>>>>>> 46cb30e77a4a9efcba6a5822359326f4473678a8
