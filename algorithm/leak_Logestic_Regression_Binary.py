import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
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
df8["Leak Alarm"] = df8["Leak Alarm"].fillna("N")
columns_to_OHE = df8[['Date', 'Leak Alarm']]
not_OHE_columns_df = df8[['ID', 'value_Lvl','value_Spr', 'Leak Found']]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(columns_to_OHE)
ohe_columns_df = pd.DataFrame(data=onehot_encoded, index=[i for i in range(onehot_encoded.shape[0])],
                      columns=['f'+str(i) for i in range(onehot_encoded.shape[1])])
preprocessed_df = ohe_columns_df.join(not_OHE_columns_df)
preprocessed_df.loc[(preprocessed_df['Leak Found'] == 'N-PRV'), 'Leak Found'] = 'N'
print(preprocessed_df)
labeled_df = preprocessed_df.loc[preprocessed_df['Leak Found'].notnull()]
unlabeled_df = preprocessed_df.loc[preprocessed_df['Leak Found'].isnull()]
shuffled_labeled_df = labeled_df.sample(frac=1).reset_index(drop=True)
labels = shuffled_labeled_df[["Leak Found"]]
# df8.to_csv('OHE.csv')
X_labeled = shuffled_labeled_df.drop(labels=['Leak Found'], axis=1) #Labled train
X_unlabeled = unlabeled_df.drop(labels=['Leak Found'], axis=1)      #Unlabled

test_ind = round(len(X_labeled)*0.70)
train_ind = test_ind + round(len(X_labeled)*0.30)
X_test = X_labeled.iloc[:test_ind]
X_train = X_labeled.iloc[test_ind:train_ind]
y_test = labels.iloc[:test_ind]
y_train = labels.iloc[test_ind:train_ind]
    # y_train['Leak Found'].value_counts().plot(kind='bar')
    # plt.xticks([0,1,2], ['No', 'Yes', 'N-PRV'])
    # plt.ylabel('Count');
    # plt.show()
    #
    # clf = LogisticRegression(max_iter=5000)
    #
    # clf.fit(X_train, y_train.values.ravel())
    # y_hat_test = clf.predict(X_test)
    # y_hat_train = clf.predict(X_train)
    #
    # train_f1 = f1_score(y_train, y_hat_train, average=None)
    # test_f1 = f1_score(y_test, y_hat_test, average=None)
    #
    # print(f"Train f1 Score: {train_f1}")
    # print(f"Test f1 Score: {test_f1}")
    #
    # plot_confusion_matrix(clf, X_test, y_test, cmap='Blues', normalize='true',
    #                      display_labels=['No.', 'Yes', 'N-PRV']);
    #
    # plt.show()

# Initiate iteration counter
iterations = 0

# Containers to hold f1_scores and # of pseudo-labels
train_f1s = []
test_f1s = []
pseudo_labels = []

# Assign value to initiate while loop
high_prob = [1]

# Loop will run until there are no more high-probability pseudo-labels
while len(high_prob) > 0:
    # Fit classifier and make train/test predictions
    # print(y_train)
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train.values.ravel())
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)

    # Calculate and print iteration # and f1 scores, and store f1 scores
    train_f1 = f1_score(y_train, y_hat_train, average='micro')
    test_f1 = f1_score(y_test, y_hat_test, average='micro')
    print(f"Iteration {iterations}")
    print(f"Train f1: {train_f1}")
    print(f"Test f1: {test_f1}")
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)

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
    # Separate predictions with > 99% probability
    high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.99],
                           df_pred_prob.loc[df_pred_prob['prob_1'] > 0.99]],
                          axis=0)
    # print(high_prob)
    print(f"{len(high_prob)} high-probability predictions added to training data.")

    pseudo_labels.append(len(high_prob))

    # Add pseudo-labeled data to training data
    X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
    high_prob = high_prob.drop(columns=['prob_0', 'prob_1'])
    print(high_prob)

    y_train = pd.concat([y_train, high_prob])
    # Drop pseudo-labeled instances from unlabeled data
    X_unlabeled = X_unlabeled.drop(index=high_prob.index)
    print(f"{len(X_unlabeled)} unlabeled instances remaining.\n")

    # Update iteration counter
    iterations += 1
    print(f"Test f1: {test_f1s}")




# def rbf_kernel_safe(X, Y=None, gamma=None):
#     X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
#     if gamma is None:
#         gamma = 1.0 / X.shape[1]
#
#     K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
#     K *= -gamma
#     K -= K.max()
#     np.exp(K, K)  # exponentiate K in-place
#     return K
#
#
# def label_prop():
#
#     labels = df9.loc[df9['Leak Found'].notnull(), ['Leak Found']]
#     model = LabelPropagation(kernel=rbf_kernel_safe)
#     model.fit(df10, labels.values.ravel())
#     pred = np.array(model.predict(df12))
#     df13 = pd.DataFrame(pred, columns=['Prediction'])
#     df14 = pd.concat([df12, df13], axis=1)
#     print(df14[['ID', 'Prediction']])
#     # print(df14.loc[df14['Prediction'] == 'Y'])
#
#
#
# if __name__ == '__main__':
#     label_prop()
