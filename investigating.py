import pandas as pd

speeches = pd.read_csv('speeches.csv', header=None)
winners = pd.read_csv('winners.csv', header=None)
freq_word = pd.read_csv('mostfreq1000word.csv', header=None, encoding='utf8', quoting=3)
freq_word_mat = pd.read_csv('mostfreq1000docword.csv', header=None)
decp_word = pd.read_csv('deceptionword.csv', header=None)
decp_word_mat = pd.read_csv('deceptiondocword.csv', header=None)
