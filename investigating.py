import pandas as pd

speeches = pd.read_csv('speeches.csv', header=None, names=['speech'])
winners = pd.read_csv('winners.csv', header=None, names=['isWin'])
freq_word = pd.read_csv('mostfreq1000word.csv', header=None, quoting=3, encoding='latin1')
freq_word_list = freq_word[0].values.tolist()
freq_word_mat = pd.read_csv('mostfreq1000docword.csv', header=None, names=freq_word_list)
decp_word = pd.read_csv('deceptionword.csv', header=None)
decp_word_list = decp_word[0].values.tolist()
decp_word_mat = pd.read_csv('deceptiondocword.csv', header=None, names=decp_word_list)

# Do stats
print(freq_word_mat['the_AT'].describe())
print(decp_word_mat['i'].describe())
print(winners['isWin'].describe())

# Do normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
def normalize(df):
  x = df.values
  x_scaled = min_max_scaler.fit_transform(x)
  return pd.DataFrame(x_scaled, columns=df.columns)

# Do stats Again
print(normalize(freq_word_mat)['the_AT'].describe())
print(normalize(decp_word_mat)['i'].describe())
print(normalize(winners)['isWin'].describe())

df = pd.concat([normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)
df.to_csv('full_data.csv')

correlation_matrix = df.corr()
correlation_matrix.to_csv('correlation_matrix.csv')

import matplotlib.pyplot as plt

plt.matshow(correlation_matrix)
plt.savefig('correlation_matrix.png')


# add speech names to df
df.insert(0, 'speech', speeches['speech'])

# add election ids to df
electionPattern = re.compile('[0-9]*[a-z]*')
df['election'] = [electionPattern.match(x).group() for x in df['speech']]

