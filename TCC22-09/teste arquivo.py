import pandas as pd

PATH = "C:/Users/johny/Downloads/bitcoin-twitter.csv/bitcoin-twitter.csv"
ENCODING = 'utf-8'

dataset= pd.read_csv(PATH,encoding = ENCODING,header = 0)

dataset.dropna (inplace=True)
print (dataset.isnull().sum())
