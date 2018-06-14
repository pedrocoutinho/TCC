import collections, re
from bagOfWords import *
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
    tso.set_count(100) #numero de tweets
    
    i=0
    with open('arquivo.txt', 'w+', encoding="utf-8") as arquivo: 
        for tweet in ts.search_tweets_iterable(tso):
            print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) ) #filtro
            #arquivo.write(tweet['user']['screen_name'] + "\n") #escrita txt
            arquivo.write(tweet['text']+"\n\n") #escrita txt
            i=i+1
except TwitterSearchException as e:
    print(e)


words = re.findall(r'\w+', open('arquivo.txt', encoding="utf8").read().lower()) #analise txt e transforma tudo para caixa baixa
c=collections.Counter(words) #contador de palavras
print (c.most_common(3)) # palavras com maior frequencia
bow.build_vocab(words) # bag of words

