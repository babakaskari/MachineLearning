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
from sklearn.semi_supervised import LabelSpreading

from gaussrank import *
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics
sns.set()


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
df8["Leak Found"] = df8["Leak Found"].fillna(-1)

dataset = df8
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
# Delete these row indexes from dataFrame
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(inplace=True)
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset1 = dataset
dataset = dataset1.drop(['Leak Alarm'], axis=1)

# ############################################################ Convert Date categorical to numerical
# dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
date_encoder = preprocessing.LabelEncoder()
date_encoder.fit(dataset['Date'])
# print(list(date_encoder.classes_))
dataset['Date'] = date_encoder.transform(dataset['Date'])
# print(dataset.to_string(max_rows=200))
dataset = dataset.drop_duplicates()
print(" dataset description : ", dataset.describe())
# ##############################################

# corrolation matrix
print(dataset.columns.values)
df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
tempdata = dataset
dataset = dataset.loc[:80]
dataset = dataset.sample(frac=1)
print("dataset shape: ", dataset.shape)
print("Number of null values in dataset : \n", dataset.isna().sum())
# print("dataset : ", dataset.shape[0])
# dataset2 = dataset.drop(["Leak Found"], axis=1)
dataset2 = dataset
print("dataset features : ", dataset.columns)
leak_found = dataset2["Leak Found"]
dataset2 = dataset.drop(['Leak Found'], axis=1)
# ########################################## APPLYING GUASSRANK NORMALIZATION

x_cols = dataset2.columns[:]
x = dataset2[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform( x )
assert x_.shape == x.shape
dataset2[x_cols] = x_
print("GaussRankScaler dataset description :\n ", dataset2.describe())

# ############################################## standard scaler
"""
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(data_scaled)
print("x_train description : ", x_train.describe())
"""
# ##############################################
print("dataset2 features : ", dataset2.columns)


x_train, x_test, y_train, y_test = train_test_split(dataset2,
                                                    leak_found,
                                                    test_size=0.2,
                                                    random_state=42)
print("y_train : ", y_train)

print('Number of data points in train data:', x_train.shape[0])
print('Number of data points in test data:', x_test.shape[0])
# print('Number of data points in test data:', x_cv.shape[0])
print(y_test)
label_prpagation_model = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=10)
label_prpagation_model.fit(x_train, y_train)
pred = label_prpagation_model.predict(x_test)
print("prediction : ", pred)
print("Result of LabelPropagation:", metrics.accuracy_score(y_test, pred))

label_spreading_model = LabelSpreading(kernel="rbf", gamma=20, max_iter=10)
label_spreading_model.fit(x_train, y_train)
pred = label_spreading_model.predict(x_test)
print("prediction : ", pred)
print("Result of LabelSpreading:", metrics.accuracy_score(y_test, pred))

error_prop_knn = []
error_spr_knn = []
error_prop_rbf = []
error_spr_rbf = []

list_k = list(range(2, 11))

for k in list_k:
    lp = LabelPropagation(kernel="knn", n_neighbors=k, max_iter=10)
    lp.fit(x_train, y_train)
    pred = lp.predict(x_test)
    error_prop_knn.append(np.mean(pred != y_test))

for k in np.arange(1.0, 15.0, 0.5):
    lp = LabelPropagation(kernel="rbf", gamma=k, max_iter=10)
    lp.fit(x_train, y_train)
    pred = lp.predict(x_test)
    error_prop_rbf.append(np.mean(pred != y_test))

for k in list_k:
    lp = LabelSpreading(kernel="knn", n_neighbors=k, max_iter=10)
    lp.fit(x_train, y_train)
    pred = lp.predict(x_test)
    error_spr_knn.append(np.mean(pred != y_test))

for k in np.arange(1.0, 15.0, 0.5):
    lp = LabelSpreading(kernel="rbf", gamma=k, max_iter=10)
    lp.fit(x_train, y_train)
    pred = lp.predict(x_test)
    error_spr_rbf.append(np.mean(pred != y_test))

# Plot sse against k
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(range(2, 11), error_prop_knn, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Error Rate vs. K Value knn Prob")
plt.xlabel("K")
plt.ylabel("Error Rate")

plt.subplot(2, 2, 2)
plt.plot(np.arange(1.0, 15.0, 0.5), error_prop_rbf, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Error Rate vs. K Value rbf Prop")
plt.xlabel("K")
plt.ylabel("Error Rate")

plt.subplot(2, 2, 3)
plt.plot(range(2, 11), error_spr_knn, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Error Rate vs. K Value knn Spr")
plt.xlabel("K")
plt.ylabel("Error Rate")

plt.subplot(2, 2, 4)
plt.plot(np.arange(1.0, 15.0, 0.5), error_spr_rbf, color="blue", linestyle="dashed", marker="o", markerfacecolor="red", markersize=10)
plt.title("Error Rate vs. K Value rbf Spr")
plt.xlabel("K")
plt.ylabel("Error Rate")

plt.show()

