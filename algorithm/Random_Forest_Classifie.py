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
df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
print("number of ID : ", df8["ID"].nunique())
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
# Delete these row indexes from dataFrame
dataset.drop(indexNames, inplace=True)
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset1 = dataset
dataset = dataset1.drop(['Leak Alarm'], axis=1)

dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
print(dataset)

print(dataset.isna().sum())
# corrolation matrix
print("Features : ")
print(dataset.columns.values)
df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
plt.show()
leak_found = dataset["Leak Found"]
dataset3 = dataset.drop(['Leak Found'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(dataset3, leak_found, stratify=leak_found, test_size=0.2)
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', x_train.shape[0])
print('Number of data points in test data:', x_test.shape[0])
print('Number of data points in test data:', x_cv.shape[0])
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Prediction : ", y_pred)
print("Score:", metrics.accuracy_score(y_test, y_pred))
print("Model score is : ", clf.score(x_test, y_test))
# print(np.mean(y_test == y_pred))
pred_probality = clf.predict_proba(x_test)
print("Predict probability : ", pred_probality)
# cross_validation score
cross_validation_score = cross_val_score(clf, x_train, y_train, cv=6)
print("Cross validation score : ", cross_validation_score)
cross_validation_predict = cross_val_predict(clf, x_train, y_train, cv=6)
print("Cross validation predict : ", cross_validation_score)
cross_val_accuracy = np.mean(cross_validation_score) * 100
print("cross validation accuracy : ", cross_val_accuracy)
# ROC
print("pred_probality : ", pred_probality, "length of prediction prob : ", len(pred_probality))
y_probs_positive = pred_probality[:, 1]
print("y_probs_positive : ", y_probs_positive)
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
print("fpr : ", fpr)
print("roc_auc_score : ", roc_auc_score(y_test, y_probs_positive))
plt.plot(fpr, tpr, color="orange", label="ROC")
plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Gussing")
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
