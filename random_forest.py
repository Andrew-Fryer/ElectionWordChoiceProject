import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from format_data import *

def score():
  df = pd.concat([elections, normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)

  X_train, X_test, y_train, y_test = train_test(df)

  clf = RandomForestClassifier().fit(X_train, y_train)

  score = clf.score(X_test, y_test)
  return score

# do cross validation
results = [score() for i in range(50)]
print(sum(results) / len(results))

