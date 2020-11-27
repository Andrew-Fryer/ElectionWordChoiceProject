import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from format_data import *

df = pd.concat([elections, normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)

def score():
  X_train, X_test, y_train, y_test = train_test_by_election(df)

  clf = DecisionTreeClassifier().fit(X_train, y_train)

  score = clf.score(X_test, y_test)
  return score

# do cross validation
results = [score() for i in range(50)]
print(sum(results) / len(results))
