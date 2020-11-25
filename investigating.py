import pandas as pd

speeches = pd.read_csv('speeches.csv', header=None, names=['speech'])
winners = pd.read_csv('winners.csv', header=None, names=['isWin'])
freq_word = pd.read_csv('mostfreq1000word.csv', header=None, quoting=3, encoding='latin1')
freq_word_list = freq_word[0].values.tolist()
freq_word_mat = pd.read_csv('mostfreq1000docword.csv', header=None, names=freq_word_list)
decp_word = pd.read_csv('deceptionword.csv', header=None)
decp_word_list = decp_word[0].values.tolist()
decp_word_mat = pd.read_csv('deceptiondocword.csv', header=None, names=decp_word_list)

df = pd.concat([speeches, freq_word_mat, decp_word_mat, winners], axis=1)
#df.to_csv('full_data.csv')

correlation_matrix = df.corr()
#correlation_matrix.to_csv('correlation_matrix.csv')

import matplotlib.pyplot as plt

plt.matshow(correlation_matrix)
#plt.savefig('correlation_matrix.png')


