from sklearn.feature_extraction.text import HashingVectorizer , TfidfVectorizer, CountVectorizer, TfidfTransformer
import dask_ml.feature_extraction.text as dask_text
from numba import njit
import setuptools
import numpy as np
import pyximport;pyximport.install(language_level = '3',
                                  setup_args={"include_dirs":np.get_include()})
import ngrams_fast as ngf
import ngrams_fast_v2 as ngf2
import cyrevhashvec as crhv
import cyhashSMARTVEC as SMTvec
import cyhashSMARTVEC_fasta as SMTvec2


import dask.bag as db
import dask.array as da
import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math


##### analyzers
import numba
from numba import njit

                
@njit 
def numbard(dataset, k):
    window = np.zeros(k)
    ret = []
    is_line = 1
    cnt = 0
    for char in dataset:
        if char == 10: 
            is_line += 1
            if is_line > 4: is_line= 1
            cnt = 0
        elif is_line == 2: 
            if char == 78: cnt = 0
            else:
                window[cnt] = char
                cnt += 1
                if cnt == k: 
                    ret.append(window)
                    cnt -= 1
                    window[:19] = window [1:20]
    return ret
        
def ngram_numba(filename, klen = 20):
    return numbard(np.fromfile(filename, dtype='int8'), klen)


def Genomes2Sparse(files = [], 
                   n_features=int(1e6), 
                   tfidf = True, 
                   binary = False,
                   MAF = 0, 
                   method = 'SMARTVEC', 
                   compute = True,
                   kmer_length = 20):
    
    if method == 'fasta_str':    
        ret = dask_text.HashingVectorizer(n_features=n_features, analyzer = lambda x: map(str,ngf.sklearn_ngram_yield_fast_old(x.encode("ASCII", errors="strict"), kmer_length))).fit_transform(db.from_sequence(files)).compute() #crhv.longRevHashPY
    elif method == 'numba_str_fastq':
        ret = dask_text.HashingVectorizer(n_features=n_features, analyzer = lambda x: numbard(x,  kmer_length)).fit_transform(db.from_sequence(files)).compute()
    elif method == 'numba':
        ret = dask_text.HashingVectorizer(n_features=n_features, analyzer = lambda x: ngram_numba(filename = x, klen = kmer_length) ).fit_transform(db.from_sequence(files)).compute()
    elif method == 'cython':
        ret = dask_text.HashingVectorizer(n_features=n_features, analyzer = lambda x: ngf.yieldkmers(filename = x, klen = kmer_length)).fit_transform(db.from_sequence(files)).compute()
    elif method == 'cython_fast':
        ret = dask_text.HashingVectorizer(n_features=n_features, analyzer = lambda x: ngf2.cyngrams(filename = x, klen = kmer_length)).fit_transform(db.from_sequence(files)).compute()
    elif method == 'cython_fastest':
        ret = dask_text.HashingVectorizer(n_features=n_features, analyzer = lambda x: ngf2.listcyngrams_fastpy(filename = x, klen = kmer_length)).fit_transform(db.from_sequence(files)).compute()
    elif method == 'reversible':
        ret = crhv.cyhashvectorizer(filenames = files, klen = kmer_length ,memorysafe= False)
    elif method == 'reversible_dd':
        b = db.from_sequence(files)
        bag2 = b.map_partitions(crhv.cyhashvectorizer, klen = kmer_length ,memorysafe= False)
        objs = bag2.to_delayed()
        arrs = [da.from_delayed(obj, (np.nan, 4**kmer_length - 4**(kmer_length//2)), str) for obj in objs]
        ret = da.concatenate(arrs, axis=0).compute()
        del arrs, objs, bag2, b
        
    elif method == 'SMARTVEC':
        b = db.from_sequence(files)
        bag2 = b.map_partitions(SMTvec2.cyhashSMARTVEC_multipleformats, klen = kmer_length ,max_size= n_features)
        objs = bag2.to_delayed()
        arrs = [da.from_delayed(obj, (1, n_features), str) for obj in objs]
        ret = da.concatenate(arrs, axis=0).compute()
        del arrs, objs, bag2, b
    #if not compute: return ret
    #ret = ret.compute()
    else:
        print('method not found')
        return -1
    if MAF > 0:
        minimum,maximum = sorted([MAF, 1-MAF])
        n_doc = len(files)
        ret = ret[:, minimum*n_doc <ret.getnnz(0)][:, ret.getnnz(0)< (maximum)*n_doc]
        if ret.shape[1] == 0: return 'no Kmer with MAF of interest'
    if tfidf: ret =  TfidfTransformer().fit_transform(ret)
    if binary: ret.data.fill(1)
    return ret
    
def binarize(X):
    return X.data.fill(1)
    
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    sns.despine()
    
def MAF_Filter_4hashed(X, AF):
    if type(AF) == int:
        minimum,maximum = sorted([AF, 1-AF])
    elif type(AF) == list:
        minimum, maximum = AF[0], AF[1]
        n_doc = X.shape[0]
    return X[:, (minimum*n_doc <X.getnnz(0)) & (X.getnnz(0)< maximum*n_doc)]
    #ret = X[:, minimum*n_doc <X.getnnz(0)]
    #return ret[:, ret.getnnz(0)< (maximum)*n_doc]

def MAF_filter_names( X, AF, vec = 0):
    if type(AF) == int:
        minimum,maximum = sorted([AF, 1-AF])
    elif type(AF) == list:
        minimum, maximum = AF[0], AF[1]
        n_doc = X.shape[0]
    if vec == 0: return np.array(range(X.shape[1]))[(minimum*n_doc <X.getnnz(0)) & (X.getnnz(0)< maximum*n_doc )]
    return vec.get_feature_names()[(minimum*n_doc <X.getnnz(0)) & (X.getnnz(0)< maximum*n_doc )]

def TFIDF(X):
    return TfidfTransformer().fit_transform(X)

def hellinger_dist_matrix(X):
    return pairwise_distances(np.sqrt(X), n_jobs=-1, metric='euclidean') / math.sqrt(2)