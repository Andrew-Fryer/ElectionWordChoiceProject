import pandas as pd
from sklearn.neural_network import MLPClassifier

from format_data import *

df = pd.concat([elections, normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)

train, test = train_test_by_election(df)

y_train = train['isWin'].values
X_train = train.drop(columns=['isWin']).values
y_test = test['isWin'].values
X_test = test.drop(columns=['isWin']).values


clf = MLPClassifier(hidden_layer_sizes=(10), max_iter=200).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
