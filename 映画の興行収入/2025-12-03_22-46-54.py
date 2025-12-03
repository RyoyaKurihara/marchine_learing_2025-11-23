
import pandas as pd

df = pd.read_csv('./cinema.csv')
df.head(10)

df2 = df.fillna(df.mean())

df2.plot(kind = 'scatter', x = 'SNS2', y = 'sales')
