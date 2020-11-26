import pandas as pd
from sklearn.neural_network import MLPClassifier

from format_data import *

# only use decp for now
df = pd.concat([elections, normalize(decp_word_mat), winners], axis=1)

train, test = train_test_by_election(df)

train = train.drop(columns=['election'])
test = test.drop(columns=['election'])

y_train = train['isWin'].values
X_train = train.drop(columns=['isWin']).values
y_test = test['isWin'].values
X_test = test.drop(columns=['isWin']).values


clf = MLPClassifier(max_iter=900).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
