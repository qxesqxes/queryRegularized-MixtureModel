#TODO tf-idf form


import numpy as np
import math
class Archive:
    def __init__(self, corpusPath=''):
        self.docList   = []
        self.backgroundLM = {}
        # self.backgroundLMArray = np.zeros(len(self.backgroundLM))
        self.docWordCountDict = {} # for PRF
        self.docLenDict = {} # length of each doc
        self.docLMDict = {}
        self.word2id = {}
        self.id2word = {} # for PRF
        self.load_ngram(corpusPath)
        self.countBackgroundLM()

    def load_ngram(self, corpusPath):
        import os
        for subdir, dirs, files in os.walk(corpusPath):
            for doc in files:
                self.docList.append(doc)
                (self.docLMDict[doc], self.docWordCountDict[doc], self.docLenDict[doc]) = self.countDocLM(os.path.join(corpusPath, doc))

    def countDocLM(self, filePath):
        fin = open(filePath, 'r')
        docWordList = list()
        for line in fin:
            lineList = line.strip().split(' ')
            docWordList.extend(lineList)
        fin.close()
        return (self.countWordListLM(docWordList), self.countWordListWordCount(docWordList), len(docWordList))

    def countWordListLM(self, wordList):
        wordCount = len(wordList)
        uniqWords = list(set(wordList))
        LM = dict()
        for word in uniqWords:
            LM[word] = float(wordList.count(word))/float(wordCount)
        return LM

    def countWordListWordCount(self, wordList):
        wordCountDict = dict()
        for word in wordList:
            wordCountDict[word] = float(wordList.count(word))
        return wordCountDict

    def dictLM2arrayLM(self, dictLM):
        arrayLM = np.zeros(len(self.backgroundLM))
        for w in dictLM:
            # since some word in query may not exist in the corpus
            if w in self.word2id:
                arrayLM[self.word2id[w]] = dictLM[w]
        return arrayLM

    def countBackgroundLM(self):
        for doc in self.docLMDict:
            for word in self.docLMDict[doc]:
                if word not in self.backgroundLM:
                    self.backgroundLM[word] = 0.0
                self.backgroundLM[word] += self.docLMDict[doc][word]

        # numpy array is more effecient format to calculate than dictionary structure
        self.backgroundLMArray = np.zeros(len(self.backgroundLM))
        for idx, word in enumerate(self.backgroundLM):
            self.backgroundLM[word] /= len(self.docList) # lazy
            self.backgroundLMArray[idx] = self.backgroundLM[word]
            self.word2id[word] = idx
            self.id2word[idx] = word

    def getMixtureLMArray(self, doc, alpha=0.4):
        docLMArray = self.dictLM2arrayLM(self.docLMDict[doc])
        return docLMArray * alpha + self.backgroundLMArray * (1-alpha)
    
    # better than interpolation mixture model
    def DirichletSmoothing(self, doc, mu=1000):
        docLMArray = self.dictLM2arrayLM(self.docLMDict[doc])
        return (self.docLenDict[doc]*docLMArray+mu*self.backgroundLMArray)/(self.docLenDict[doc]+mu)





