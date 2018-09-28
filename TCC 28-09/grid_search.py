"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from preprocessor1 import Preprocessor 
import pandas as pd


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
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=0)

#declara o vetorizador - 3
vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=True, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)
    
#vetorização dos textos - (Passa como parametro apenas os textos)
X_train     = vectorizer.fit_transform(X_train)
X_test      = vectorizer.transform(X_test)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
