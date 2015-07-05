#TODO random walk
#TODO cosine similarity

import numpy as np
from Archive import Archive
import argparse, os
from math import log
import multiprocessing

class QueryRegularizedMixtureModel:
    def __init__(self, archiveObj, queryStr, qid, topN, mu, firstPassDirPath, queryExpansionStr=None, expansionWeight=0.2):
        self.archive    = archiveObj    # the document corpus object
        self.topN       = topN          # the number of pseudo relevant documents
        self.mu         = mu            # Both EM parameters and Dirichlet smoothing parameters
        self.qid = qid                  # the query id

        if queryExpansionStr is None:
            print('Query Regularized Mixture Model')
            self.queryLMArray = self.countQueryLMArray(queryStr)
        else:
            print('Query Regularized Mixture Model with Query Expansion')
            self.queryLMArray = self.countQueryExpansionLMArray(queryStr, queryExpansionStr, expansionWeight)            

        self.pseudoReldocList = self.firstPassRetrieval(topN, firstPassDirPath)

        self.RelLMArray = np.zeros(len(self.queryLMArray))
        self.alpha = {}
        self.initRelLMArray()

    def __mkdir(self, dirPath):
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

    # query expansion version
    def countQueryExpansionLMArray(self, queryStr, queryExpansionStr, expansionWeight):
        queryList = queryStr.strip().split(' ')
        expansionList = queryExpansionStr.strip().split(' ')
        queryLM = self.archive.countWordListLM(queryList)
        expansionLM = self.archive.countWordListLM(expansionList)
        return (1-expansionWeight)*self.archive.dictLM2arrayLM(queryLM) + expansionWeight*self.archive.dictLM2arrayLM(expansionLM)

    # no query expansion version
    def countQueryLMArray(self, queryStr):
        queryList = queryStr.strip().split(' ')
        queryLM = self.archive.countWordListLM(queryList)
        return self.archive.dictLM2arrayLM(queryLM)
        
    def getArrayNonZeroIdx(self, array):
        return [i for i in range(len(array)) if array[i]>0.0]

    def printQueryInfo(self, nonZeroIdx):
        outputStr=''
        for i in nonZeroIdx:
            outputStr += self.archive.id2word[i]+':'+str(self.queryLMArray[i])+' '
        print('Query ['+str(self.qid)+'] = '+outputStr)

    def firstPassRetrieval(self, topN, outdir=None):
        nonZeroIdx_query = self.getArrayNonZeroIdx(self.queryLMArray)
        self.printQueryInfo(nonZeroIdx_query)

        doc_score = [] # tuple list 
        for idx, doc in enumerate(self.archive.docList):
            mixtureLMArray = self.archive.DirichletSmoothing(doc, self.mu)
            score = self.KLDivergence(self.queryLMArray, mixtureLMArray, nonZeroIdx_query)
            doc_score.append((doc, score))
        doc_score = sorted(doc_score, key=lambda obj: obj[1], reverse=True)  

        if outdir != None:
            self.__mkdir(outdir)
            self.dumpRetrievalResults(doc_score, outdir)

        ranked_doc = [pair[0] for pair in doc_score]
        return ranked_doc[:topN]

    # nonZeroIdx speed up the process of counting KLD x60
    def KLDivergence(self, queryLMArray, docLMArray, nonZeroIdx):
        kld = 0.0
        for i in nonZeroIdx:
            kld += queryLMArray[i]*log(docLMArray[i])
        return kld

    def initRelLMArray(self):
        for doc in self.pseudoReldocList:
            self.RelLMArray += self.archive.DirichletSmoothing(doc)
            self.alpha[doc] = 0.5
        self.RelLMArray /= float(len(self.pseudoReldocList))

    def EMTraining(self, iter):
        for i in range(iter):
            self.calEtable()
            self.calMstep()

    # calculate E step
    def calEtable(self):
        self.ETable = {}
        RnonZeroIdx = self.getArrayNonZeroIdx(self.RelLMArray)
        for i in RnonZeroIdx:
            self.ETable[i] = {}
            for doc in self.pseudoReldocList:
                alpha_doc = self.alpha[doc]
                self.ETable[i][doc] = alpha_doc * self.RelLMArray[i] / (alpha_doc * self.RelLMArray[i] + (1 - alpha_doc) * self.archive.backgroundLMArray[i])

    # calculate M step
    def calMstep(self):
        # count for alphas
        for doc in self.alpha:
            self.alpha[doc] = 0.0
            for w in self.archive.docLMDict[doc]:
                try:
                    self.alpha[doc] += self.ETable[self.archive.word2id[w]][doc] * self.archive.docLMDict[doc][w]
                except:
                    self.alpha[doc] += 0

        # count RelLMArray
        RnonZeroIdx = self.getArrayNonZeroIdx(self.RelLMArray)
        denominator = 0.0
        for wordidx in self.ETable:
            for doc in self.pseudoReldocList:
                denominator += self.ETable[wordidx][doc] * self.archive.docWordCountDict[doc].get(self.archive.id2word[wordidx], 0.0)
        denominator += self.mu

        for i in RnonZeroIdx:
            numerator = 0.0
            for doc in self.ETable[i]:
                numerator += self.ETable[i][doc] * self.archive.docWordCountDict[doc].get(self.archive.id2word[i], 0.0)
            numerator += self.mu * self.queryLMArray[i]
            self.RelLMArray[i] = numerator / denominator

    def dumpRetrievalResults(self, doc_score, outdir):
        fo = open('%s/%i' % (outdir, self.qid), 'w')
        fo.write('\n'.join(['%i 0 %s 0 %.6f %s' % (self.qid, pair[0], pair[1], 'EXP') for pair in doc_score]))
        fo.write('\n')
        fo.close()

    def secondPassRetrieval(self, outdir=None): 
        doc_score = []
        nonZeroIdx_RelLM = self.getArrayNonZeroIdx(self.RelLMArray)
        for doc in self.archive.docList:
            mixtureLMArray = self.archive.DirichletSmoothing(doc, self.mu)
            score = self.KLDivergence(self.RelLMArray, mixtureLMArray, nonZeroIdx_RelLM)
            doc_score.append((doc, score))
        doc_score = sorted(doc_score, key=lambda obj: obj[1], reverse=True)

        if outdir != None:
            self.__mkdir(outdir)
            self.dumpRetrievalResults(doc_score, outdir)

        ranked_docDict = dict()
        for pair in doc_score:
            ranked_docDict[pair[0]] = pair[1]
        return ranked_docDict

def algo_noExpansion(archive, queryStr, qid, pseudoDocNum, mu, firstPassResultDir, secondPassResultDir):
    prf = QueryRegularizedMixtureModel(archive, queryStr, qid, pseudoDocNum, mu, firstPassResultDir)
    prf.EMTraining(iter = 2)
    prf.secondPassRetrieval(outdir=secondPassResultDir)

def algo_expansion(archive, queryStr, qid, pseudoDocNum, mu, firstPassResultDir, secondPassResultDir, expansionStr, expansionWeight):
    prf = QueryRegularizedMixtureModel(archive, queryStr, qid, pseudoDocNum, mu, firstPassResultDir, expansionStr, expansionWeight)
    prf.EMTraining(iter = 2)
    prf.secondPassRetrieval(outdir=secondPassResultDir)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('queryList', help='the word-segmented query list.')
    parser.add_argument('corpusPath', help='the path of corpus including all documents')
    parser.add_argument('firstPassResultDir', help='the first pass result directory')
    parser.add_argument('secondPassResultDir', help='the second pass result directory')
    parser.add_argument('-n', '--pseudoDocNum', help='the number of pseudo relevant docs', type=int, default=10)
    parser.add_argument('-m', '--mu', help='the Dirichlet smoothing parameter', type=float, default=1000)
    parser.add_argument('-e', '--queryExpansion', help='the query expansion list, no expansion just ignore')
    parser.add_argument('-w', '--expansionWeight', help='the query expansion weight', type=float, default=0.2)
    args = parser.parse_args()

    # multiprocessing
    pool = multiprocessing.Pool(processes=6)

    # process the corpus 
    print('Processing the corpus ...')
    archive = Archive(args.corpusPath)

    # retrieve algorithm
    if args.queryExpansion is None: 
        fin_query = open(args.queryList, 'r')
        for qid, queryStr in enumerate(fin_query): 
            pool.apply_async(algo_noExpansion, (archive, queryStr, qid+1, args.pseudoDocNum, args.mu, args.firstPassResultDir, args.secondPassResultDir,))
            #prf = QueryRegularizedMixtureModel(archive, queryStr, qid+1, args.pseudoDocNum, 1000, args.firstPassResultDir)
    else:
        fin_query = open(args.queryList, 'r')
        fin_expansion = open(args.queryExpansion, 'r')
        for qid,(queryStr, expansionStr) in enumerate(zip(fin_query, fin_expansion)):
            pool.apply_async(algo_expansion, (archive, queryStr, qid+1, args.pseudoDocNum, args.mu, args.firstPassResultDir, args.secondPassResultDir, expansionStr, args.expansionWeight))
            #prf = QueryRegularizedMixtureModel(archive, queryStr, qid+1, args.pseudoDocNum, 1000, args.firstPassResultDir, expansionStr, args.expansionWeight)

    pool.close()
    pool.join()
    print('Jobs done!')

