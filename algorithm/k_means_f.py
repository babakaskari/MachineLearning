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
import prepossessed_dataset
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import test
from gaussrank import *
from sklearn.preprocessing import StandardScaler

sns.set()

pd.set_option('mode.chained_assignment', None)
dataset = prepossessed_dataset.unlabeled()
x_train = dataset["x_train"]
x_test = dataset["x_test"]
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

