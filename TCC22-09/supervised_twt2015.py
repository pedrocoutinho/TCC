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
import itertools
import shutil
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import  f1_score


from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import Preprocessor 



def benchmark():
    
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        clf_test()
        
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
    
        
        


def clf_test():
        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
    
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                                target_names=categories))
        
        
        
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
        
        return metrics.f1_score(y_test, pred, average='weighted')  
        

###############################################################################
#####################__Parâmetros pra rodar o algoritmo__######################
###############################################################################
clf = SVC(C=13, kernel='linear', decision_function_shape='ovo')#clf=svm.NuSVC(nu=0.7,kernel = 'linear',probability = True,class_weight = 'balanced',)
#clf=LinearSVC(loss='squared_hinge', penalty='l2',dual=False, tol=1e-3, class_weight='balanced')
NUM_QUESTIONS = 1
ACTIVE = 1
PATH_TRAIN = "C:/Users/Seven/Documents/tcc_active learning/dados/twt_2015/2015train.csv"
PATH_TEST  = "C:/Users/Seven/Documents/tcc_active learning/dados/twt_2015/2015test.csv"
ENCODING = 'utf-8'
result_x = []
result_y = []
###############################################################################






###############################################################################
################__Carrega dados dos arquivos + Preprocessamento__##############
###############################################################################        
 
df_train = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0)
df_test = pd.read_csv(PATH_TEST,encoding = ENCODING,header = 0)


df_neg = df_train[df_train.pol == 'negative'].reset_index(drop=True)
df_pos = df_train[df_train.pol == 'positive'].reset_index(drop=True)
df_nlt = df_train[df_train.pol == 'neutral'].reset_index(drop=True)



df_train = pd.DataFrame(df_neg[0:df_neg.size])
df_train = df_train.append(df_pos[0:df_pos.size], ignore_index=True)
df_train = df_train.append(df_nlt[0:df_nlt.size], ignore_index=True)
df_train = df_train.reindex(np.random.permutation(df_train.index))
df_train = df_train.reset_index(drop=True)


df_unlabeled = pd.DataFrame(df_neg[2:df_neg.size])
df_unlabeled = df_unlabeled.append(df_pos[2:df_pos.size], ignore_index=True)
df_unlabeled = df_unlabeled.append(df_nlt[2:df_nlt.size], ignore_index=True)
df_unlabeled.rename(columns={'text':'bruto'}, inplace=True)
df_unlabeled = df_unlabeled.reindex(np.random.permutation(df_unlabeled.index))
df_unlabeled = df_unlabeled.reset_index(drop=True)

    
categories = df_train.pol.unique()
       
pp = Preprocessor()
df_train.text   = pp.preprocess(df_train.text)
df_test.text    = pp.preprocess(df_test.text)
df_unlabeled    = df_unlabeled.assign(text = pp.preprocess(df_unlabeled.bruto))
###############################################################################

        



###############################################################################
##########################-->Loop do active learning<--########################
############################################################################### 
while True:
    
    print(df_train.pol.size)
    print(df_unlabeled.pol.size)
    print(df_test.pol.size)        
    y_train = df_train.pol
    y_test =  df_test.pol
    

    
    vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=True, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)
    
    
    X_train     = vectorizer.fit_transform(df_train.text)
    X_test      = vectorizer.transform(df_test.text)
    X_unlabeled = vectorizer.transform(df_unlabeled.text)
    
    
    
    
    benchmark()    
    result_x.append(clf_test())
    result_y.append(df_train.pol.size)      
    
        
    if df_unlabeled.pol.size > 3:
        insert = {'pol':[],'text':[]}
        cont = 0
        for i in range(0, 12):
           insert["pol"].insert(cont,df_unlabeled.pol[i])
           insert["text"].insert(cont,df_unlabeled.text[i]) 
           cont+=1
           df_unlabeled = df_unlabeled.drop(i)
            
        
        df_insert = pd.DataFrame.from_dict(insert)
        df_train= df_train.append(df_insert, ignore_index=True)
        df_unlabeled = df_unlabeled.reset_index(drop=True)
        
        #labelNumber = input("Aperte qualquer tecla para próxima iteração")
        
    else:
        
        
        result_y_2015_spv = result_y
        result_x_2015_spv = result_x
        plt.plot(result_y_2015_spv, result_x_2015_spv)
        plt.axis([0, 480, 0.2, 0.6])
        plt.grid(True)
        plt.show()
        
        result = pd.DataFrame(result_y)
        result = result.assign(y=result_x)
        np.savetxt(r'C:/Users/Seven/Documents/tcc_active learning/resultados/2013_active.txt', result, fmt='%f')
        break
