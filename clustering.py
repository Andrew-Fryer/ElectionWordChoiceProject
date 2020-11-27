import pandas as pd
from sklearn.neural_network import MLPClassifier

from format_data import *


df = pd.concat([normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)
df.to_csv('full_data.csv')

correlation_matrix = df.corr()
correlation_matrix.to_csv('correlation_matrix.csv')

import matplotlib.pyplot as plt

plt.matshow(correlation_matrix)
plt.savefig('correlation_matrix.png')


df = pd.concat([elections, normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)

X_train, X_test, y_train, y_test = train_test_by_election(df)

clf = MLPClassifier(hidden_layer_sizes=(10), max_iter=200).fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)
