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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
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
# corrolation matrix
# print(dataset.columns.values)
# df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
# print(dataset.loc[dataset['Leak Found'].isna()])
# print("tempdata : \n ", tempdata.shape[0])
x_train = dataset.loc[dataset['Leak Found'].isna()]
x_train = x_train.drop(["Leak Found"], axis=1)

# x_train = x_train.sample(frac=1)
# print("x_train shape : ", x_train.shape)
x_test = dataset.loc[dataset['Leak Found'].notna()]
y_test = x_test.loc[dataset['Leak Found'].notna(), ['Leak Found']]
print("y_test is equal to : ", y_test)
x_test = x_test.drop(["Leak Found"], axis=1)
x_centroid = np.array(x_test.loc[16:17])
print(x_centroid.shape)
print("x_train : \n", x_train)
print("x_test : \n ", x_test)

# score = make_scorer(acc, greater_is_better=False)

attributes = {
    'n_clusters': [2],
    'init': ['k-means++', 'random'],
    'random_state': [None, 0],
    'max_iter': [300, 600, 1200],
    # 'max_iter': np.arange(100, 2000, 100),
    'algorithm': ['full', 'auto']

}
kmeansModel = KMeans()

gs_k_means = GridSearchCV(estimator=kmeansModel,
                          param_grid=attributes,
                          cv=5,
                          n_jobs=-1,
                          verbose=2,)

gs_k_means.fit(x_train)
print("gs_k_means best parameter is : ", gs_k_means.best_params_)
print("gs_k_means score : ", gs_k_means.score(x_test))
y_pred = gs_k_means.predict(x_test)
print("Prediction : ", y_pred)



