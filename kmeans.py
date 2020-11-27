import pandas as pd

from format_data import *

df = pd.concat([normalize(freq_word_mat), normalize(decp_word_mat)], axis=1) # try with election data too

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=num_elections).fit(df.values)
print('cluster centers:', kmeans.cluster_centers_)

# let's see if the clusters correspond to the 14 elections
print('these are the elections that are mapped to each cluster:')
clusters = [[] for i in range(num_elections)]
for i in range(len(kmeans.labels_)):
  cluster_number = kmeans.labels_[i]
  clusters[cluster_number - 1].append(elections['election'][i])
for i in range(len(clusters)):
  print(clusters[i])
print()


