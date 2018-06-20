import collections, re, nltk
from bagOfWords3 import *
from TwitterSearch import *

try:
 
    ts = TwitterSearch(
        consumer_key = 'IMJh4kjQLGDzUaT9t1v0RXm5Y',
        consumer_secret = 'cjt9d684CpvElXof1BxUMgSakNnFBVLDweQTSpGZolzzrnU8JE',
        access_token = '968521944944529408-oI5NcJVaZellwrsPjhsQkQPDeAZJzKf',
        access_token_secret = 'hc7bTI65fG97smD3ZEB6iCjLrBzHBxn2Sp6TIaX8fZSJZ'
     )
 
    tso = TwitterSearchOrder()
    tso.set_keywords(['bitcoin']) #parametros de pesquisa
    tso.set_language('pt') # idioma
    tso.set_count(5) #numero de tweets
    stemmer = nltk.stem.RSLPStemmer()

    i=0
    with open('arquivo.txt', 'w+', encoding="utf-8") as arquivo: 
        for tweet in ts.search_tweets_iterable(tso):
            remove_links = bow.clean_tweet(tweet['text']) 
            arquivo.write(remove_links+"\n\n") #escrita txt
            i=i+1
except TwitterSearchException as e:
    print(e)


words = re.findall(r'\w+', open('arquivo.txt', encoding="utf8").read().lower()) #analise txt e transforma tudo para caixa baixa
inputs = bow.build_vocab(words) # bag of words
words2 = re.findall(r'\w+', open('stopWords.txt', encoding="utf8").read())
listafinal = list(set(inputs)- set(words2))
bow.word_stemmer(listafinal)



