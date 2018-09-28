from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from preprocessor1 import Preprocessor 
from sklearn import metrics

PATH_TRAIN = "C:/git/TCC/TCC22-09/BTC_tweets_daily_66.csv"
PATH_TEST  = "C:/git/TCC/TCC22-09/BTC_tweets_daily_34.csv"
ENCODING = 'utf-8'
	
df_train = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0) #importar o csv
df_test = pd.read_csv(PATH_TEST,encoding = ENCODING,header = 0)


pp = Preprocessor()
df_train.Tweet = pp.preprocess(df_train.Tweet)
df_test.Tweet = pp.preprocess(df_test.Tweet)

categories = df_train.Sentiment.unique()

vectorizerNB= CountVectorizer(analyzer="word")  
X_train     = vectorizerNB.fit_transform(df_train.Tweet)
X_test      = vectorizerNB.transform(df_test.Tweet)


Y_train = df_train.Sentiment 
Y_test = df_test.Sentiment

model = MultinomialNB()
model.fit(X_train, Y_train)
teste=model.predict(X_test)

print(metrics.classification_report(Y_test, teste,
                                                target_names=categories))
print(metrics.confusion_matrix(Y_test, teste))
print (metrics.f1_score(Y_test, teste, average='weighted'))
