import pandas as pd

from format_data import *

df = pd.concat([normalize(freq_word_mat), normalize(decp_word_mat)], axis=1) # try with election data too

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.000001, min_samples=2).fit(df.values)

# let's see if the clusters correspond to the 14 elections
print('these are the elections that are mapped to each cluster:')
clusters = [[] for i in range(len(set(db.labels_)))]
for i in range(len(db.labels_)):
  cluster_number = db.labels_[i]
  clusters[cluster_number - 1].append(elections['election'][i])
for i in range(len(clusters)):
  print(clusters[i])
