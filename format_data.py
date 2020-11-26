import pandas as pd

speeches = pd.read_csv('speeches.csv', header=None, names=['speech'])
winners = pd.read_csv('winners.csv', header=None, names=['isWin'])
freq_word = pd.read_csv('mostfreq1000word.csv', header=None, quoting=3, encoding='latin1')
freq_word_list = freq_word[0].values.tolist()
freq_word_mat = pd.read_csv('mostfreq1000docword.csv', header=None, names=freq_word_list)
decp_word = pd.read_csv('deceptionword.csv', header=None)
decp_word_list = decp_word[0].values.tolist()
decp_word_mat = pd.read_csv('deceptiondocword.csv', header=None, names=decp_word_list)

# Do normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
def normalize(df):
  x = df.values
  x_scaled = min_max_scaler.fit_transform(x)
  return pd.DataFrame(x_scaled, columns=df.columns)

# pull election ids from speech names
import re
electionPattern = re.compile('[0-9]*[a-z]*')
elections = pd.DataFrame([electionPattern.match(x).group() for x in speeches['speech']], columns=['election'])

# split data into training and testing data
import random
def train_test_by_election(df):
  # first we will split the 14 election results into 10 for training and 4 for testing
  testing_elections = []
  unique_elections = list(set(elections['election'].values))
  for i in range(4):
    choosen = unique_elections[random.randint(0, len(unique_elections) - 1)]
    unique_elections.remove(choosen)
    testing_elections.append(choosen)
  training_elections = unique_elections
  # now divide the entire dataset according that split
  train = df[df['election'].isin(training_elections)].drop(columns=['election'])
  test = df[df['election'].isin(testing_elections)].drop(columns=['election'])
  return train, test
