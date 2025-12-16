
'''
# 行 × 列の調べる

データの量を確認する。
'''

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('./Survived.csv')

df.shape

'''
Out[10]: (891, 11)
'''