
# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
# from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as cgb
from sklearn.ensemble import RandomTreesEmbedding

import warnings

warnings.filterwarnings('ignore')

# preprocessing

# ## ###########################
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
# ##################################################### CORRELATION MATRIX
# print(dataset.columns.values)
# dataset2 = dataset.drop(["Leak Found"], axis=1)
# df = pd.DataFrame(dataset2, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
# ########################################################################


def evaluate_preds(model, x_true, y_true, y_preds):
    accuracy = metrics.accuracy_score(y_true, y_preds)
    precision = metrics.precision_score(y_true, y_preds)
    recall = metrics.recall_score(y_true, y_preds)
    f1 = metrics.f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print("Model score is : ", model.score(x_true, y_true))
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision : {precision: .2f}")
    print(f"Recall : {recall: .2f}")
    print(f"F1 Score : {f1: .2f}")

    return metric_dict

# ##################################################### SPLIT THE DATASET


x_labeled_data = dataset.loc[dataset['Leak Found'].notna()]
y_labeled_date = x_labeled_data["Leak Found"]
x_labeled_data = x_labeled_data.drop(["Leak Found"], axis=1)
# ############################################## standard scaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x_labeled_data)
x_train = pd.DataFrame(data_scaled)
print("x_train after normalization :\n ", x_train.head())
print("x_train description after normalization:\n ", x_labeled_data.describe())


# ############################################################
# x_train = x_train.sample(frac=1)
x_unlabeled_data = dataset.loc[dataset['Leak Found'].isna()]
y_unlabeled_data = x_unlabeled_data.drop(["Leak Found"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_labeled_data,
                                                    y_labeled_date,
                                                    test_size=0.2,
                                                    random_state=42)
x_train, x_cv, y_train, y_cv = train_test_split(x_train,
                                                y_train,
                                                stratify=y_train,
                                                test_size=0.2)




"""
model_factory = [
    LogisticRegression(),
    KNeighborsClassifier(),
    BaggingClassifier(n_estimators=100),
    XGBRegressor(nthread=1),
    GradientBoostingRegressor(),
    RandomForestClassifier(),
    BayesianRidge(),
]
"""

estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rfc', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('adab', AdaBoostClassifier(n_estimators=100, random_state=0)),
    ('gb', GradientBoostingClassifier()),
    ('bc', BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)),
    ('etc', ExtraTreesClassifier()),
    ('hgbc', HistGradientBoostingClassifier()),
    ('xgb', XGBClassifier())
    ]


model_factory = [
    KNeighborsClassifier(),
    LogisticRegression(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
   # RandomTreesEmbedding(),
    StackingClassifier(estimators=estimators),
    VotingClassifier(estimators=estimators),
    HistGradientBoostingClassifier(),
    XGBClassifier(),

]

# #######################################  XGBClassifier()
clf_xgb = XGBClassifier()
clf_xgb.fit(x_train, y_train)
xgb_pred = clf_xgb.predict(x_test)
xgb_matrices = evaluate_preds(clf_xgb, x_test, y_test, xgb_pred)

# #################################################################
# #######################################  KNeighborsClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(x_train, y_train)
knn_pred = clf_knn.predict(x_test)
knn_matrices = evaluate_preds(clf_knn, x_test, y_test, knn_pred)

# #################################################################
# ################################################ AdaBoostClassifier starts
clf_adab = AdaBoostClassifier(n_estimators=100, random_state=0)
clf_adab.fit(x_train, y_train)
adab_pred = clf_adab.predict(x_test)
adab_matrices = evaluate_preds(clf_adab, x_test, y_test, adab_pred)
# ################################################ AdaBoostClassifier ends
# ################################################ RandomForestClassifier
clf_rfc = RandomForestClassifier()
clf_rfc.fit(x_train, y_train)
rfc_pred = clf_rfc.predict(x_test)
rfc_matrices = evaluate_preds(clf_rfc, x_test, y_test, rfc_pred)
# ############################################################
# ################################################ GradientBoostingClassifier
clf_gbc = GradientBoostingClassifier()
clf_gbc.fit(x_train, y_train)
clf_pred = clf_gbc.predict(x_test)
gbc_matrices = evaluate_preds(clf_gbc, x_test, y_test, clf_pred)
# ############################################################
# ############################################################ BaggingClassifier
clf_bc = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)
clf_bc.fit(x_train, y_train)
bc_pred = clf_bc.predict(x_test)
bc_matrices = evaluate_preds(clf_bc, x_test, y_test, bc_pred)
# ################################################ ExtraTreesClassifier
clf_etc = ExtraTreesClassifier()
clf_etc.fit(x_train, y_train)
etc_pred = clf_etc.predict(x_test)
et_matrices = evaluate_preds(clf_etc, x_test, y_test, etc_pred)
# ############################################################
# ############################################################ HistGradientBoostingClassifier
clf_hgbc = HistGradientBoostingClassifier()
clf_hgbc.fit(x_train, y_train)
hgbc_pred = clf_hgbc.predict(x_test)
hgb_matrices = evaluate_preds(clf_hgbc, x_test, y_test, hgbc_pred)
# ############################################################
# ############################################################ LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(x_train, y_train)
clf_pred = clf_lr.predict(x_test)
lr_matrices = evaluate_preds(clf_lr, x_test, y_test, clf_pred)
# ############################################################
# ############################################################ StackingClassifier
clf_sc = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf_sc.fit(x_train, y_train)
clf_pred = clf_sc.predict(x_test)
sc_matrices = evaluate_preds(clf_sc, x_test, y_test, clf_pred)
# ############################################################
# ############################################################   VotingClassifier
clf_vc = VotingClassifier(estimators=[
                            ("knn", clf_knn),
                            ('adab', clf_adab),
                            ('rfc', clf_rfc),
                            ('gnc', clf_gbc),
                            ("bc", clf_bc),
                            ("etc", clf_etc),
                            ("hgbc", clf_hgbc),
                            ('xgb', clf_xgb),
                            ("lr", clf_lr)], voting='soft')

clf_vc.fit(x_train, y_train)
clf_pred = clf_vc.predict(x_test)
vc_matrices = evaluate_preds(clf_vc, x_test, y_test, clf_pred)

# ############################################################


compare_matrices = pd.DataFrame({
                                "KNeighbors": knn_matrices,
                                "RandomForest": rfc_matrices,
                                "AdaBoost" : adab_matrices,
                                "GradientBoos": gbc_matrices,
                                "Bagging": bc_matrices,
                                "ExtraTrees": et_matrices,
                                "HistGradientBoosting": hgb_matrices,
                                "LogisticRegression": lr_matrices,
                                "StackingClassifier": sc_matrices,
                                "VotingClassifier": vc_matrices,
                                "XGBClassifier": xgb_matrices,
                                 })

compare_matrices.plot.bar(rot=0)
plt.show()



