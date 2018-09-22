import numpy as np
import re, nltk

class BagOfWords():
    def __init__(self):
         self.vocab = [] #Construtor de lista de palavras

    def build_vocab(self, sentences):
        for sentence in sentences:    #pega sentença do grupo sentenças, até acabar as sentenças
            for word in sentence.split(' '): #pega palavra por palavra da sentença até acabar a sentença
                    if word not in self.vocab: # se a palavra não estiver no vocabulário ele add
                            self.vocab.append(word)

        inputs = self.vocab  
        return inputs                  
        #print(self.vocab) #printa o vocabulário todo

    def toarray(self, sentence):   #tipo da sentença, são grupos de palavras separados por espaço
        words = sentence.split(' ')
       # print(words)

    def clean_tweet(self, tweet):
        tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
        tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
        tweet = re.sub('#\S+', '', tweet)  # remove hashtags
        tweet = re.sub('@\S+', '', tweet)  # remove mentions
        tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', tweet)  # remove punctuations
        tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
        return tweet

    def word_stemmer(self, listafinal):
        with open('bow.txt', 'w+', encoding="utf-8") as bow:
            for i in listafinal:
                radical = stemmer.stem(i)
                print("'" + radical+"'"+" ")
                bow.write("'" + radical+"'"+"\n")
            

inputs = []
listafinal = []
tweet = '0' 
stemmer = nltk.stem.RSLPStemmer()
bow = BagOfWords()
bow.build_vocab(inputs) 
bow.clean_tweet(tweet)
bow.word_stemmer(listafinal)

