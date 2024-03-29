import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
#df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
#df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
############################################################# Delete these row indexes from dataFrame
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(drop=True, inplace=True)
############################################################# DROPPING LEAK ALARM & LEAK FOUND
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset = dataset.drop(['Leak Alarm'], axis=1)

############################################################# Convert Date categorical to numerical
# dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
date_encoder = preprocessing.LabelEncoder()
date_encoder.fit(dataset['Date'])
# print(list(date_encoder.classes_))
dataset['Date'] = date_encoder.transform(dataset['Date'])
# print(dataset.to_string(max_rows=200))
print("Number of null values in dataset :\n", dataset.isna().sum())
############################################################ SPLIT THE DATASET INTO LABELED AND UNLABELED
labeled_df = dataset.loc[dataset['Leak Found'].notnull()]
unlabeled_df = dataset.loc[dataset['Leak Found'].isnull()]
shuffled_labeled_df = labeled_df.sample(frac=1).reset_index(drop=True)
labels = shuffled_labeled_df[["Leak Found"]]
############################################################ APPLYING GUASSRANK NORMALIZATION
# x_cols = x_train.columns[:]
# x = x_train[x_cols]
#
# s = GaussRankScaler()
# x_ = s.fit_transform( x )
# assert x_.shape == x.shape
# x_train[x_cols] = x_
# ############################################################ TO REPRESENT OUR DATASET, ALL COLUMNS IN MATRIX FORM
# x_train = pd.DataFrame(x_train)
# pd.plotting.scatter_matrix(x_train)
# x_train.plot(kind='density', subplots=True, sharex=False)
# plt.show()

X_labeled = shuffled_labeled_df.drop(labels=['Leak Found'], axis=1)   ###################  Labled train
X_unlabeled = unlabeled_df.drop(labels=['Leak Found'], axis=1)        ###################  Unlabled
# ########################################################### APPLYING GUASSRANK NORMALIZATION
# =======================labeled data
x_cols = X_labeled.columns[:]
x = X_labeled[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform( x )
assert x_.shape == x.shape
X_labeled[x_cols] = x_

# ===================== unlabeled data

x_cols = X_unlabeled.columns[:]
x = X_unlabeled[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform( x )
assert x_.shape == x.shape
X_unlabeled[x_cols] = x_

# #################################################
# ############################################################ SPLIT
y_train = labels
X_train = X_labeled
print("Number of null values in X_train :\n", X_train.isna().sum())
print("Number of null values in y_train :\n", y_train.isna().sum())

############################################################INITIATE INTERACTION COUNTER
iterations = 0

###########################################################Containers to hold f1_scores and # of pseudo-labels
train_f1s = []
test_f1s = []
pseudo_labels = []

############################################################Assign value to initiate while loop
high_prob = [1]
############################################################Loop will run until there are no more high-probability pseudo-labels
while len(X_unlabeled) > 0:
    # Fit classifier and make train/test predictions
    # print(y_train)

    clf = LogisticRegression(max_iter=100)
    # clf.fit(X_train, y_train.values.ravel())
    clf.fit(X_train, y_train.values.ravel())
    # Generate predictions and probabilities for unlabeled data
    print(f"Now predicting labels for unlabeled data...")

    pred_probs = clf.predict_proba(X_unlabeled)  
    preds = clf.predict(X_unlabeled)
    prob_0 = pred_probs[:, 0]
    prob_1 = pred_probs[:, 1]
    # Store predictions and probabilities in dataframe
    df_pred_prob = pd.DataFrame([])
    df_pred_prob['Leak Found'] = preds
    df_pred_prob['prob_0'] = prob_0
    df_pred_prob['prob_1'] = prob_1
    df_pred_prob.index = X_unlabeled.index
    # print("df_pred_prob : ", df_pred_prob)
    # Separate predictions with > 99% probability
    high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.99],
                           df_pred_prob.loc[df_pred_prob['prob_1'] > 0.99]],
                          axis=0)
    # print("High prob : ", high_prob)
    # print(high_prob)
    print(f"{len(high_prob)} high-probability predictions added to training data.")

    if not high_prob.empty:        
        pseudo_labels.append(len(high_prob))
    # Add pseudo-labeled data to training data
        X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)    
        high_prob = high_prob.drop(columns=['prob_0', 'prob_1']) 
        frames = [y_train, high_prob]
        result = pd.concat(frames, ignore_index=False)
        result = pd.concat([y_train, high_prob], axis=0, ignore_index=False)
        y_train = pd.concat([y_train, high_prob])
        X_unlabeled = X_unlabeled.drop(index=high_prob.index)

    # print("y_train : ", y_train)
    # unshuffeled_df = pd.concat([X_train, y_train["Leak Found"]], axis=1)
    unshuffeled_df = pd.concat([X_train, y_train], axis=1)
    shuffeled_df = unshuffeled_df.sample(frac=1)    
    y_train['Leak Found'] = shuffeled_df['Leak Found']    
    # print("y_train shuffeled_df: ", y_train)
    # print("Number of null values in y_train :\n", y_train.isna().sum())
    X_train = shuffeled_df.drop(['Leak Found'], axis=1)
    # print("Number of null values in X_train :\n", X_train.isna().sum())
    
    # Drop pseudo-labeled instances from unlabeled data
    
    print(f"{len(X_unlabeled)} unlabeled instances remaining.\n")

    # Update iteration counter
    iterations += 1
    print("iterations : ", iterations)
    print(f"Test f1: {test_f1s}")

print("SHuffeled data : ", shuffeled_df)





