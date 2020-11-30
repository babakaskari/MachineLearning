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
import sys
import seaborn as sns
sns.set()

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("./dataset/Acoustic Logger Data.csv")
df1 = df.loc[df["LvlSpr"] == "Lvl"]
df3 = df.loc[df["LvlSpr"] == "Spr"]
df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df5 = pd.merge(df2, df4, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df6 = df5.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df6['Date'] = pd.to_datetime(df6['Date'], format='%d-%b')
df6['Date'] = df6['Date'].dt.strftime('%d-%m')
pathh = sys.path[0] + "\dataset"
df7 = pd.read_csv("./dataset/Leak Alarm Results.csv")
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
dataset1 = dataset
dataset = dataset1.drop(['Leak Alarm'], axis=1)
print("dataset : ", dataset)
print(dataset.isna().sum())
# corrolation matrix
print("Features : ", dataset.columns.values)

""""
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

corr = dataset[["value_Lvl", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[0, 0])

corr = dataset[["value_Spr", "Leak Found"]].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[0, 1])

# corr = dataset[["Leak Alarm", "Leak Found"]].corr()
# print(corr)
# sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axs[1, 0])

plt.show()
"""

df = pd.DataFrame(dataset, columns=['ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
print("ssytem path : ", sys.path)
print(sys.path[0] + "\dataset")
""""
y_pred = clf.predict(x_test)
print("Prediction : ", y_pred)
print("Score:", metrics.accuracy_score(y_test, y_pred))
print("Model score is : ", clf.score(x_test, y_test))
# print(np.mean(y_test == y_pred))
pred_probality = clf.predict_proba(x_test)
print("Predict probability : ", pred_probality)
# cross_validation score
cross_validation_score = cross_val_score(clf, x_train, y_train, cv=5)
print("Cross validation score : ", cross_validation_score)
cross_validation_predict = cross_val_predict(clf, x_train, y_train, cv=5)
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
"""
"""
x_train, x_test, y_train, y_test = train_test_split(dataset2,
                                                    leak_found,
                                                    stratify=leak_found,
                                                    test_size=0.2,
                                                    random_state=42)
# x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
print('Number of data points in train data:', x_train.shape[0])
print('Number of data points in test data:', x_test.shape[0])
# print('Number of data points in test data:', x_cv.shape[0])

"""