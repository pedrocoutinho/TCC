import pandas as pd
from sklearn.svm import SVC
from preprocessor1 import Preprocessor 
from sklearn.feature_extraction.text import TfidfVectorizer


'''
- Define parametros
- Importa dados
'''
clf = SVC(C=13, kernel='linear', decision_function_shape='ovo')
PATH_TRAIN = "C:/Users/Pedro/Desktop/TCC/HPython/BTC_tweets3.csv"
#PATH_TEST  = "C:/Users/Seven/Documents/tcc_active learning/dados/twt_2015/2015test.csv"
ENCODING = 'utf-8'


df_train = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0) #importar o csv
pp = Preprocessor()
df_train.Tweet = pp.preprocess(df_train.Tweet)
print(df_train.Tweet)

 #declara o vetorizador - 3
vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=True, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)
    
 #vetorização dos textos - (Passa como parametro apenas os textos)
X_train     = vectorizer.fit_transform(df_train.Tweet)
print('VETORIZAÇÃO')
print(X_train)



