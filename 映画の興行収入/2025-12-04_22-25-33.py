
import pandas as pd

df = pd.read_csv('./cinema.csv')
df.head(10)

df2 = df.fillna(df.mean())

no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index

df3 = df2.drop(no, axis=0)

x = df3[['SNS1','SNS2','actor','original']] #特徴量
t = df3['sales'] #正解データ


