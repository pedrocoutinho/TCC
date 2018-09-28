import pandas as pd
from time import time
from sklearn import metrics, model_selection, linear_model
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor1 import Preprocessor 
from sklearn.naive_bayes import MultinomialNB

def benchmark():
    
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        #treina o modelo
        clf.fit(x_train, y_train)         
        #para usar o classificador uma vez treinado com a função fit, usa-se a função predict
        clf_test() 
        
        
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
    
def clf_test():
        t0 = time()
        pred = clf.predict(x_test)  
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
    
        print("classification report:")
        print(metrics.classification_report(y_test, pred, target_names=categories))
        
        
        
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
        
        return metrics.f1_score(y_test, pred, average='weighted')  
        
'''
- Define parametros
- Importa dados
'''
PATH = "C:/Users/Pedro/Desktop/TCC/HPython/TESTE/BTC_tweets_daily.csv"
ENCODING = 'utf-8'

#importa o arquivo csv
dataset = pd.read_csv(PATH,encoding = ENCODING,header = 0) 

#declara as categorias
categories = dataset.Sentiment.unique()

#Preprocessa os tweets
pp = Preprocessor()
dataset.Tweet = pp.preprocess(dataset.Tweet)

# Carrega os elementos do dataset
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Split em conjunto de treino e teste
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.34, random_state=0)

#declara o vetorizador - 3
vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=True, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)
    
#vetorização dos textos - (Passa como parametro apenas os textos)
x_train     = vectorizer.fit_transform(x_train)
x_test      = vectorizer.transform(x_test)

#declara o Classificador
opt = input('1- SVM'+ '\n' + '2- Rede Neural' + '\n' + '3- Nayve Bayes' + '\n')

if opt == '1':
        #CLASSIFICADOR SVC
        clf = SVC(C=13, kernel='linear', decision_function_shape='ovo')
elif opt == '2':
        #CLASSIFICADOR REDE NEURAL
        clf = linear_model.Perceptron(max_iter=1000)
elif opt == '3':
        #CLASSIFICADOR NAYVE BAYES
        clf = MultinomialNB()

        
benchmark()



