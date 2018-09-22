import numpy as np
import re

class BagOfWords():
    def __init__(self):
         self.vocab = [] #Construtor de lista de palavras

    def build_vocab(self, sentences):
        for sentence in sentences:    #pega sentença do grupo sentenças, até acabar as sentenças
            for word in sentence.split(' '): #pega palavra por palavra da sentença até acabar a sentença
                    if word not in self.vocab: # se a palavra não estiver no vocabulário ele add
                            self.vocab.append(word)
        print(self.vocab) #printa o vocabulário todo

    def toarray(self, sentence):   #tipo da sentença, são grupos de palavras separados por espaço
        words = sentence.split(' ')
        print(words)
        #bagofwords.write(words+"\n") #escrita txt

    def removeStopwords(self,wordlist, stopwords):
        #with open('teste2.txt', 'w+', encoding="utf-8") as arquivo:
        for w in wordlist: 
            if w in wordlist:
                print (w + "\n")  


#stopwords = open('stopWords.txt', encoding="utf8")
inputs = ['eu','quero','isso','ai']

words2 = re.findall(r'\w+', open('stopWords.txt', encoding="utf8").read())
 
bow = BagOfWords()
bow.build_vocab(inputs) 
#print (words2)
bow.removeStopwords(inputs, words2)
listafinal= list(set(inputs)- set(words2))
print (listafinal)