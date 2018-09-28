'''
import pandas as pd
from sklearn.svm import SVC
from preprocessor1 import Preprocessor 
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn import metrics
from sklearn.metrics import  f1_score
'''
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import os
from time import time
import numpy as np
import pylab as pl
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
import itertools
import shutil
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import  f1_score
from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessor1 import Preprocessor 


def benchmark(): # Naive Bayes
	
	X_train     = vectorizerNB.fit_transform(df_train.Tweet)
	X_test      = vectorizerNB.transform(df_test.Tweet)

	model = MultinomialNB()
	model.fit(X_train, Y_train)
	print(clf)

	teste=model.predict(X_test)
	print (teste)


'''
def benchmark(): #SVM
    
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, Y_train) #treina o modelo        
        clf_test() 
        #para usar o classificador uma vez treinado com a função fit, usa-se a função predict
        
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
  
  '''  
def clf_test():
        t0 = time()
        pred = clf.predict(X_test) # 
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
    
        print("classification report:")
        print(metrics.classification_report(Y_test, pred,
                                                target_names=categories))
        
        
        
        print("confusion matrix:")
        print(metrics.confusion_matrix(Y_test, pred))
        
        return metrics.f1_score(Y_test, pred, average='weighted')  
        
'''
- Define parametros
- Importa dados
'''
clf = SVC(C=13, kernel='linear', decision_function_shape='ovo')
PATH_TRAIN = "C:/git/TCC/TCC22-09/BTC_tweets_daily_66.csv"
PATH_TEST  = "C:/git/TCC/TCC22-09/BTC_tweets_daily_34.csv"
ENCODING = 'utf-8'


df_train = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0) #importar o csv
df_test = pd.read_csv(PATH_TEST,encoding = ENCODING,header = 0)

categories = df_train.Sentiment.unique()

pp = Preprocessor()
df_train.Tweet = pp.preprocess(df_train.Tweet)
df_test.Tweet = pp.preprocess(df_test.Tweet)


#polaridades de todos os tweets
Y_train = df_train.Sentiment 
Y_test = df_test.Sentiment

#declara o vetorizador - 3
vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=True, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)
vectorizerNB= CountVectorizer(analyzer="word")  

#vetorização dos textos - (Passa como parametro apenas os textos)
X_train     = vectorizer.fit_transform(df_train.Tweet)
X_test      = vectorizer.transform(df_test.Tweet)


benchmark()



