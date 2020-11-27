import pandas as pd
from sklearn.neural_network import MLPClassifier

from format_data import *

# Do stats
print(freq_word_mat['the_AT'].describe())
print(decp_word_mat['i'].describe())
print(winners['isWin'].describe())

# Do stats Again on normalized data
print(normalize(freq_word_mat)['the_AT'].describe())
print(normalize(decp_word_mat)['i'].describe())
print(normalize(winners)['isWin'].describe())
