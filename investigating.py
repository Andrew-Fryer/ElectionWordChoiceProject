import pandas as pd

speeches = pd.read_csv('speeches.csv')
freq_word = pd.read_csv('mostfreq1000word.csv', encoding='utf8', quoting=3)
freq_word_mat = pd.read_csv('mostfreq1000docword.csv')
decp_word = pd.read_csv('deceptionword.csv')
decp_word = pd.read_csv('deceptiondocword.csv')
