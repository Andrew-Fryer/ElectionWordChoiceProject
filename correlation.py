import pandas as pd

from format_data import *

df = pd.concat([normalize(freq_word_mat), normalize(decp_word_mat), winners], axis=1)

correlation_matrix = df.corr()

import matplotlib.pyplot as plt

plt.matshow(correlation_matrix)
plt.savefig('correlation_matrix.png')

