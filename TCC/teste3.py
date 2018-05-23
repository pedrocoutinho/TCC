
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
    tso.set_language('pt') #linguagem
    tso.set_count(100) #numero de tweets
    
    with open('arquivo.txt', 'w+', encoding="utf-8") as arquivo: 
        for tweet in ts.search_tweets_iterable(tso):
            print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) ) #filtro
            arquivo.write(tweet['user']['screen_name'] + "\n") #escrita txt
            arquivo.write(tweet['text']+ "\n\n") #escrita txt

except TwitterSearchException as e:
    print(e)
