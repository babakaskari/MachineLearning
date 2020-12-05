import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from gaussrank import *
from sklearn.preprocessing import StandardScaler

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
###################################################### Delete these row indexes from dataFrame
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(drop=True, inplace=True)
###################################################### DROPPING LEAK ALARM & LEAK FOUND
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset = dataset.drop(['Leak Alarm'], axis=1)

# ############################################################ Convert Date categorical to numerical
# dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
date_encoder = preprocessing.LabelEncoder()
date_encoder.fit(dataset['Date'])
# print(list(date_encoder.classes_))
dataset['Date'] = date_encoder.transform(dataset['Date'])
# print(dataset.to_string(max_rows=200))
print("Number of null values in dataset :\n", dataset.isna().sum())
###################################################### CORRELATION MATRIX
# print(dataset.columns.values)
# dataset2 = dataset.drop(["Leak Found"], axis=1)
# df = pd.DataFrame(dataset2, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
###################################################### SPLIT THE DATASET
x_train = dataset.loc[dataset['Leak Found'].isna()]
x_train = x_train.drop(["Leak Found"], axis=1)
# x_train = x_train.sample(frac=1)
x_test = dataset.loc[dataset['Leak Found'].notna()]
y_test = x_test.loc[dataset['Leak Found'].notna(), ['Leak Found']]
###################################################### CORRELATION OF KNOWN LABELLED DATA
df = pd.DataFrame(x_test, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
######################################################
x_test = x_test.drop(["Leak Found"], axis=1)
print("x_test shape is equal to :  ", x_test.shape)
print("dataset features :  ", dataset.columns)
############################################## CREATING DUMMY_DATA
# x_centroid = np.array(x_test.iloc[[16, 17], ])
dummy_data = dataset.drop(['Leak Found'], axis=1)
print("Description  : \n ", dummy_data.describe())
############################################## TO TAKE THE SELECTED SAMPLE FOR OUR XTRAIN
# dummy_data = dummy_data.sample(frac=1)
#x_dummy = dummy_data[:54]
#x_train = x_dummy
############################################# SCALER NORMALIZATION   " TO BE MODIFIED LATER"

# scaler = MinMaxScaler()
# # fit using the train set
# scaler.fit(x_train)
# # transform the test test
# x_train = scaler.transform(x_train)
# # build the scaler model
# scaler = Normalizer()
#
# # fit using the train set
# scaler.fit(x_train)
# # transform the test test
# x_train = scaler.transform(x_train)
# plt.show()
########################################### APPLYING GUASSRANK NORMALIZATION
"""
x_cols = x_train.columns[:]
x = x_train[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform( x )
assert x_.shape == x.shape
x_train[x_cols] = x_
"""
############################################### standard scaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(data_scaled)
print("x_train description : ", x_train.describe())
########################################## TO REPRESENT OUR DATASET, ALL COLUMNS IN MATRIX FORM
x_train = pd.DataFrame(x_train)
pd.plotting.scatter_matrix(x_train)
x_train.plot(kind='density', subplots=True, sharex=False)
plt.show()
############################################ APPLYING KMEANS
kmeans = KMeans(n_clusters=2,
                init="k-means++",
                random_state=None,
                max_iter=300,
                algorithm='auto',
                n_init=1000).fit(x_train)
y_pred = kmeans.predict(x_train)
############################################ TO JUST SHOW PREDICTION AND ID BY SCATTER
# dataframe = pd.DataFrame(y_pred, columns=['y-pred'])
# dataframe["ID"]= x_test["ID"]
# plt.scatter(x_test["ID"], dataframe["y-pred"], label='skitscat', color='k', s=25, marker="o")
# plt.show()
############################################ TO GET THE CENTROID AND PRINT OUT
centroids = kmeans.cluster_centers_
print("Cluster centroids are : \n", centroids)
X = np.arange(1, len(x_train) + 1)
# print("x_train shape", x_train.shape)
print("x_test features  :  ", x_test.columns)
# plt.scatter(x_train["Date"], x_train["value_Lvl"], s=50)
# plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
# plt.show()
############################################# ACCURACY
print("Prediction : \n ", y_pred)
# print(metrics.accuracy_score(y_test, y_pred))
############################################# TO GET THE FINAL SCATTER AFTER PREDICTION
d_numpy = x_train.to_numpy()

plt.scatter(d_numpy[y_pred == 0, 2], d_numpy[y_pred == 0, 3], s=25, c='green', label='Cluster 1')
plt.scatter(d_numpy[y_pred == 1, 2], d_numpy[y_pred == 1, 3], s=25, c='blue', label='Cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', label='Two assumed centroid points')
plt.title('Clusters')
plt.legend()
plt.show()
# Run the Kmeans algorithm and get the index of data points clusters
elbo = []
list_k = list(range(2, 11))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(x_train)
    elbo.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, elbo, '-o')
plt.xlabel(r'Number of clusters')
plt.ylabel('Sum of squared distance')
plt.show()

