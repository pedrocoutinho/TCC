import numpy as np
class BagOfWords:
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

inputs = []

bow = BagOfWords() 
bow.build_vocab(inputs)