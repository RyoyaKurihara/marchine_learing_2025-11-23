
# 各列のnullをカウント
# 各列の欠損値を確認する。

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('./Survived.csv')

df.isnull().sum()
