import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve
from sklearn.semi_supervised import LabelPropagation
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn import metrics
sns.set()

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
df1 = df.loc[df["LvlSpr"] == "Lvl"]
df3 = df.loc[df["LvlSpr"] == "Spr"]
df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df5 = pd.merge(df2, df4, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df6 = df5.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df6['Date'] = pd.to_datetime(df6['Date'], format='%d-%b')
df6['Date'] = df6['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("../dataset/Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df7 = df7.rename(columns={'Date Visited': 'Date'})

df8 = pd.merge(df6, df7, on=['ID', 'Date'], how='left')
df8 = df8.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
# df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
# df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
# Delete these row indexes from dataFrame
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(inplace=True)
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset1 = dataset
dataset = dataset1.drop(['Leak Alarm'], axis=1)
df_date = dataset["Date"]
print("df_date describe : ", df_date.describe())
dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
print("Number of null values in dataset :\n", dataset.isna().sum())
# corrolation matrix
print(dataset.columns.values)
df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
data_pre = dataset
# data_pre.drop(data_pre[data_pre['value_Lvl'] > 200].index, inplace=True)
print("unique  : ", data_pre['Date'].nunique())
print("dataset describe : ", data_pre.describe())
X = np.arange(1, len(data_pre) + 1)
# plt.scatter(data_pre['value_Lvl'], data_pre['Leak Found'], alpha=0.5)
plt.scatter(data_pre['Date'], data_pre['value_Spr'], alpha=0.5, c='r')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Scatter value_Lvl')
# plt.legend()
# plt.scatter(data_pre['value_Spr'], alpha=0.5, c='b')
tempdata = dataset.drop(["Leak Found"], axis=1)
print("tempdata : ", tempdata.shape[0])
x_train = tempdata.loc[56:]
x_train = x_train.sample(frac=1)
# print("x_train shape : ", x_train.shape)
x_test = tempdata.loc[3002:3010]

print("x_train : ", x_train)
print("x_test : ", x_test)

kmeans = KMeans(n_clusters=2, init='k-means++',  max_iter=300, random_state=0, algorithm='auto').fit(x_train)
centroids = kmeans.cluster_centers_
print("Cluster centroids are : ", centroids)
# plt.scatter(x_train['value_Lvl'], x_train['value_Spr'], alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
y_pred = kmeans.predict(x_test)
print("Prediction : ", y_pred)


