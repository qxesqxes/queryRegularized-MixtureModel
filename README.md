queryRegularized-MixtureModel
=============================

Query-regularized Mixture Model

Provide an implementation of Query-Regularized Mixture Model in Tao, Tao, and ChengXiang Zhai. "Regularized estimation of mixture models for robust pseudo-relevance feedback." Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 2006.


Some Default
------------------
The default dump format is TREC\_eval

Multi-processing thread = 6


Requirement
------------------
python numpy

Usage
-------
python queryRegularizedMixtureModel.py -h

usage: queryRegularizedMixtureModel.py [-h] [-n PSEUDODOCNUM] [-m MU]
[-e QUERYEXPANSION]
[-w EXPANSIONWEIGHT]
queryList corpusPath firstPassResultDir
secondPassResultDir

positional arguments:
queryList             the word-segmented query list.
corpusPath            the path of corpus including all documents
firstPassResultDir    the first pass result directory
secondPassResultDir   the second pass result directory

optional arguments:
-h, --help            show this help message and exit
-n PSEUDODOCNUM, --pseudoDocNum PSEUDODOCNUM
the number of pseudo relevant docs
-m MU, --mu MU        the Dirichlet smoothing parameter
-e QUERYEXPANSION, --queryExpansion QUERYEXPANSION
the query expansion list, no expansion just ignore
-w EXPANSIONWEIGHT, --expansionWeight EXPANSIONWEIGHT
the query expansion weight


