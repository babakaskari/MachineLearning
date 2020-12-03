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
#df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
# Delete these row indexes from dataFrame
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)

# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)

dataset = dataset.drop(['Leak Alarm'], axis=1)

dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
# print(dataset.to_string(max_rows=200))
print("Number of null values in dataset :\n", dataset.isna().sum())
# # corrolation matrix
# print(dataset.columns.values)
# df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
# print(dataset.loc[dataset['Leak Found'].isna()])
# print("tempdata : \n ", dataset.shape[0])

x_train = dataset.loc[dataset['Leak Found'].isna()]
x_train = x_train.drop(["Leak Found"], axis=1)

# x_train = x_train.sample(frac=1)
# print("x_train shape : ", x_train.shape)
x_test = dataset.loc[dataset['Leak Found'].notna()]
y_test = x_test.loc[dataset['Leak Found'].notna(), ['Leak Found']]

# print(y_test)

df = pd.DataFrame(x_test, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()

x_test = x_test.drop(["Leak Found"], axis=1)
print("x_test shape is equal to : ", x_test.shape)
x_centroid = np.array(x_test.iloc[[16, 17], ])
# print(x_centroid)
# print("x_train : \n", x_train)
# print("x_test : \n ", x_test)
print("dataset features : ", dataset.columns)
dummy_data = dataset.drop(['Leak Found'], axis=1)
dummy_data = dummy_data.sample(frac=1)
x_dummy = dummy_data[0:3000]
x_train = dummy_data

kmeans = KMeans(n_clusters=2,
                init="k-means++",
                random_state=None,
                max_iter=300,
                algorithm='auto',
                n_init=1000).fit(x_train)
y_pred = kmeans.predict(x_test)
centroids = kmeans.cluster_centers_
print("Cluster centroids are : ", centroids)
X = np.arange(1, len(x_train) + 1)
# print("x_train shape", x_train.shape)
print("x_test features  : ", x_test.columns)
# plt.scatter(x_train["Date"], x_train["value_Lvl"], s=50)
# plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
# plt.show()
print("Prediction : \n ", y_pred)
print(metrics.accuracy_score(y_test, y_pred))
y_kmeans = kmeans.fit_predict(x_train)
print("k_means predict is equal to : ", y_kmeans)
d_numpy = x_train.to_numpy()
plt.scatter(d_numpy[y_kmeans == 0, 0], d_numpy[y_kmeans == 0, 1], s=100, c='green', label='Cluster 1')
plt.scatter(d_numpy[y_kmeans == 1, 0], d_numpy[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
plt.title('Clusters')
plt.legend()
plt.show()

