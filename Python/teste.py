import oauth2
import json
import pprint

consumer_key = 'YgIPEThUD6Qz2e224NA9hXZrz'
consumer_secret = 'NwpG9iw1pwHYA3wFFP0AzeJpIt2jSmj7vqrixOMBPW2rNedvYh'

token_key = '972606605677223936-oD43YoUw8w1CgUN4SANVE1anS1UzeXb'
token_secret = 'JegjESJQ9akdK8YjvPHLbiuF2PH3yjtKJPSwf30ejbJZn'

consumer = oauth2.Consumer(consumer_key, consumer_secret)
token = oauth2.Token (token_key, token_secret)
cliente = oauth2.Client(consumer, token)

requisicao = cliente.request('https://api.twitter.com/1.1/search/tweets.json?q=flamengo')

decodificar = requisicao[1].decode()

objeto = json.loads(decodificar)
twittes = objeto['statuses']
teste = 2*twittes

i=0
for twit in teste:
	print(twit['user']['screen_name'])
	print(twit['text'])
	i = i+1
	print()

print(i)