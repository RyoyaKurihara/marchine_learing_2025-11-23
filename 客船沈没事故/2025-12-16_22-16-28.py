
# 学習用データの確認

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv('./Survived.csv')
df.head(2) # 先頭2行の確認


