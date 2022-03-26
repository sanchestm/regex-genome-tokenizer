import dask.array as da
import dask.bag as db
from dask.dataframe import Series
import dask.dataframe as dd
import warnings
from sklearn.feature_extraction.text import TfidfTransformer
from dask.distributed import Client, TimeoutError
import scipy as sp
import scipy.sparse as sps
from numba.typed import Dict, List
from numba import prange
from numba.types import unicode_type
from scipy.sparse import csr_matrix
import pandas as pd
import glob
import re
import io
import numpy as np
import dask.dataframe as dd
import numba as nb
from glob import glob
import pyximport;pyximport.install(language_level = '3',
                                  setup_args={"include_dirs":np.get_include()})
import cythongenomeanalyzer as cyGA
import umap
import time
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import math

##### basic numba functions

@nb.njit(fastmath = True)
def char2int(chars):
    if chars == 65: return 0
    elif chars == 67: return 1
    elif chars == 71: return 2
    elif chars == 84: return 3
    return -1

@nb.njit(fastmath = True)
def invchar2int(chars):
    if chars == 65: return 3
    elif chars == 67: return 2
    elif chars == 71: return 1
    elif chars == 84: return 0
    return -1

@nb.njit(fastmath = True)
def str2int(chars):
    if chars == 'A' or chars == 'a': return 0
    elif chars == 'C' or chars == 'c': return 1
    elif chars == 'G' or chars == 'g': return 2
    elif chars == 'T' or chars == 't': return 3
    return -1

@nb.njit(fastmath = True)
def invstr2int(chars):
    if chars == 'A' or chars == 'a': return 3
    elif chars == 'C' or chars == 'c': return 2
    elif chars == 'G' or chars == 'g': return 1
    elif chars == 'T' or chars == 't': return 0
    return -1


@nb.njit(fastmath = True)
def _numba_array_fastq(file_mmap, klen, max_size = -1, mg_fw_rv = True): 
    if mg_fw_rv: array_size = 4**klen - 4**(klen//2)
    else: array_size = 4**klen 
    fw_div = 4**(klen-1)
    if max_size > 0 and max_size < array_size:
        array_size = max_size
    is_line = 1 
    window_pos = 0
    fw_num = 0
    rv_num = 0
    if mg_fw_rv == True:
        for c in file_mmap:
            if c == 78: window_pos, fw_num, rv_num = 0,0,0
            elif c == 13: pass
            elif c == 10: 
                is_line += 1
                if is_line > 4: is_line= 1
                window_pos = 0
            elif is_line == 2: 
                fw_num = fw_num*4 + char2int(c)
                rv_num = invchar2int(c)*(4**window_pos) + rv_num
                window_pos += 1
                if window_pos == klen: 
                    if fw_num < rv_num: result[fw_num%array_size] += 1
                    else: result[rv_num%array_size] += 1
                    fw_num %= fw_div
                    rv_num //= 4
                    window_pos -= 1
    else:
        for c in file_mmap:
            if c == 78: window_pos, fw_num = 0,0
            elif c == 13: pass
            elif c == 10: 
                is_line += 1
                if is_line > 4: is_line= 1
                window_pos = 0
            elif is_line == 2: 
                fw_num = fw_num*4 + char2int(c)
                window_pos += 1
                if window_pos == klen: 
                    result[fw_num%array_size] += 1
                    fw_num %= fw_div
                    window_pos -= 1
    return result

@nb.njit(fastmath = True)
def _numba_array_fasta(file_mmap, klen, max_size = -1, mg_fw_rv = True): 
    if mg_fw_rv: array_size = 4**klen - 4**(klen//2)
    else: array_size = 4**klen 
    fw_div = 4**(klen-1)
    if max_size > 0 and max_size < array_size: array_size = max_size
    window_pos = 0
    fw_num = 0
    rv_num = 0
    result = np.zeros((array_size), dtype = np.int16)
    
    header_end = 0
    if file_mmap[0] == 62: 
        while file_mmap[header_end] != 10: header_end+=1 
            
    if mg_fw_rv == True:
        for c in file_mmap[header_end:]:
            if c == 78: window_pos, fw_num, rv_num = 0,0,0
            elif c == 10 or c == 13: pass
            else: 
                fw_num = fw_num*4 + char2int(c)
                rv_num = invchar2int(c)*(4**window_pos) + rv_num
                window_pos += 1
                if window_pos == klen: 
                    if fw_num < rv_num: result[fw_num%array_size] += 1
                    else: result[rv_num%array_size] += 1
                    fw_num %= fw_div
                    rv_num //= 4
                    window_pos -= 1
    else:
        for c in file_mmap[header_end:]:
            if c == 78: window_pos, fw_num = 0,0
            elif c == 10 or c == 13: pass
            else: 
                fw_num = fw_num*4 + char2int(c)
                window_pos += 1
                if window_pos == klen: 
                    result[fw_num%array_size] += 1
                    fw_num %= fw_div
                    window_pos -= 1
    return result

@nb.njit(fastmath = True)
def _numba_string(string, klen, max_size = -1, mg_fw_rv = True): 
    if mg_fw_rv: array_size = 4**klen - 4**(klen//2)
    else: array_size = 4**klen 
    fw_div = 4**(klen-1)
    if max_size > 0 and max_size < array_size:
        array_size = max_size
    window_pos = 0
    fw_num = 0
    rv_num = 0
    result = np.zeros((array_size), dtype = np.int16)
    if mg_fw_rv:
        for c in string:
            if c == 'N': window_pos, fw_num, rv_num = 0,0,0
            elif c == '\n': pass
            else: 
                fw_num = fw_num*4 + str2int(c)
                rv_num = invstr2int(c)*(4**window_pos) + rv_num
                window_pos += 1
                if window_pos == klen: 
                    if fw_num < rv_num: result[fw_num%array_size] += 1
                    else: result[rv_num%array_size] += 1
                    fw_num %= fw_div
                    rv_num //= 4
                    window_pos -= 1
    else:
        for c in string:
            if c == 'N': window_pos, fw_num = 0,0
            elif c == '\n': pass
            else: 
                fw_num = fw_num*4 + str2int(c)
                window_pos += 1
                if window_pos == klen: 
                    result[fw_num%array_size] += 1
                    fw_num %= fw_div
                    window_pos -= 1
    return result

#@nb.jit
def nb_multiple_files(filenames = [], klen = 10, max_size = -1 , cnt = 'auto', merge_reverse_complement = True):
    if cnt == 'auto':
        typ = filenames[0].split('.')[-1]
        if typ == filenames[0]: typ = 'string'
    else: typ = cnt    
    if typ == 'fastq': 
        ret = sps.vstack((sps.csr_matrix(_numba_array_fastq(np.memmap(name,  mode='r'), klen, max_size, merge_reverse_complement)) for  name  in filenames))
    elif typ == 'fasta' or typ == 'fa':
        ret = sps.vstack((sps.csr_matrix(_numba_array_fasta(np.memmap(name,  mode='r'), klen, max_size,merge_reverse_complement)) for  name  in filenames))
    elif typ == 'string':
        ret = sps.vstack((sps.csr_matrix(_numba_string(cont, klen, max_size, merge_reverse_complement)) for  cont  in filenames))
    else:
        ret = sps.eye(0, dtype = np.int16 ,format="csr")
    ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
    return ret

def nb_sparse_fasta(filenames = [], klen = 10, max_size = -1 , merge_reverse_complement = True):
    ret = sps.vstack((sps.csr_matrix(_numba_array_fasta(np.memmap(name,  mode='r'), klen, max_size,merge_reverse_complement)) for  name  in filenames))
    ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
    return ret

def nb_sparse_string(filenames = [], klen = 10, max_size = -1 , merge_reverse_complement = True):
    ret = sps.vstack((sps.csr_matrix(_numba_string(cont, klen, max_size, merge_reverse_complement)) for  cont  in filenames))
    ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
    return ret

def nb_sparse_fastq(filenames = [], klen = 10, max_size = -1 , merge_reverse_complement = True):
    ret = sps.vstack((sps.csr_matrix(_numba_array_fastq(np.memmap(name,  mode='r'), klen, max_size, merge_reverse_complement)) for  name  in filenames))
    ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
    return ret

class KmerGenomeAnalyser:
    
    def __init__(self,minorallelefrequency = 0, kmer_size = 10, MergeReverseComplement = False, highmemory = False):
        if minorallelefrequency:
            if type(minorallelefrequency) == int or type(minorallelefrequency) == float:
                self.maf = sorted([minorallelefrequency, 1-minorallelefrequency])
            elif type(minorallelefrequency) == list or type(minorallelefrequency) == tuple:
                self.maf = sorted([minorallelefrequency[0], minorallelefrequency[1]])
            else:
                raise ValueError('Minor Allele Fequency not accepted')
        else: self.maf =0
        
        if kmer_size < 0:
            raise ValueError('Kmer size has to be positive')
        if kmer_size > 14 and not highmemory: 
            ValueError('Each array will take more than 1gb per individual')
        else: self.kmer = kmer_size
        
        self.revComp = MergeReverseComplement
        self.n_docs = 0
                
    def nbG2S(self, 
              files = [], 
               n_features=int(1e7), 
               tfidf = False, 
               binary_output = False,
               content = 'auto', 
               single_row_output = False):

        ncols = min(n_features, 4**self.kmer - 4**(self.kmer//2) if self.revComp else 4**self.kmer)
        if ncols == n_features: 
            warnings.warn("Not all kmers will be accounted for, if kmer index > n_features we will only use the suffix") 
        if ncols > 1e9: raise ValueError('Each array will take more than 1gb per individual')

        if content == 'auto':
            if (typ := files[0].split('.')[-1]) == files[0]: typ = 'string'
        else: typ = content
        
            
        b = db.from_sequence(files)
        if typ == 'cython':
            bag2 = b.map_partitions(cyGA.cyfiles2sparse, klen = self.kmer ,max_size= n_features, merge_reverse_complement = self.revComp )
        elif typ == 'fastq':
            bag2 = b.map_partitions(self.nb_sparse_fastq, klen = self.kmer ,max_size= n_features, merge_reverse_complement = self.revComp )
        elif typ == 'fasta' or typ == 'fa':
            bag2 = b.map_partitions(self.nb_sparse_fasta, klen = self.kmer ,max_size= n_features, merge_reverse_complement = self.revComp )
        elif typ == 'string':
            bag2 = b.map_partitions(self.nb_sparse_string, klen = self.kmer ,max_size= n_features, merge_reverse_complement = self.revComp )
        else: return 'content type not found'

        objs = bag2.to_delayed()
        arrs = [da.from_delayed(obj, (np.nan, ncols), str) for obj in objs]
        ret = da.concatenate(arrs, axis=0).compute()
                
        if self.maf:
                n_doc = ret.shape[0]
                self.ogshape = (self.maf[0]*n_doc <ret.getnnz(0)) & (ret.getnnz(0)< self.maf[1]*n_doc)
                ret = ret[:,self.ogshape]
                if ret.shape[1] == 0: raise ValueError('No kmers present with minor allele frequency range sent')
        else:
            self.ogshape = (True for x in range(ret.shape[1]))
        
        if tfidf: ret =  TfidfTransformer().fit_transform(ret)
        
        if binary_output: ret.data.fill(1)
            
        if single_row_output: ret = sps.csr_matrix((ret.data, ret.indices, (0, ret.indptr[-1])), dtype = np.int16, copy=False, shape = (1, ncols))
        
        del arrs, objs, bag2, b
        
        return ret
    
    def applymaf(X, maf):
        if maf:
            warnings.warn("kmer IDs will be lost if <class>.maf is not altered too") 
            if type(maf) == int or type(maf) == float:
                mafed = sorted([maf, 1-maf])
            elif type(maf) == list or type(maf) == tuple:
                mafed = sorted([maf[0], maf[1]])
            else:
                raise ValueError('Minor Allele Fequency not accepted')
            n_doc = X.shape[0]
            X = X[:,(mafed[0]*n_doc <X.getnnz(0)) & (X.getnnz(0)< mafed[1]*n_doc)]
            if X.shape[1] == 0: raise ValueError('No kmers present with minor allele frequency range sent')
        return X
        

    def int2kmer(self, i):
        q = np.base_repr(i,4).replace('0', 'A').replace('1', 'C').replace('2', 'G').replace('3', 'T')
        return q if len(q) == self.kmer else ''.join('A' for i in range(self.kmer - len(q))) + q

    def kmer_list(self):
        return [self.int2kmer(i) for i, accepted in enumerate(self.ogshape) if accepted]
    
    def indices2kmers(self, indices):
        return [self.int2kmer(i) for i in self.new_indexes2old_indexes(np.array(indices), self.ogshape)]
    
    @staticmethod
    @nb.njit(fastmath = True)
    def new_indexes2old_indexes(index_array , ogshape):
        output = np.zeros(len(index_array), dtype = np.uint32) 
        k = -1
        j = 0
        a_index = 0
        for i in ogshape:
            k += i
            if k == index_array[a_index]: 
                output[a_index]  = j
                if a_index == len(index_array)-1: break
                else: a_index += 1
            j += 1
        return output

    @staticmethod
    @nb.njit(fastmath = True)
    def str_to_int(s):
        final_index, result = len(s) - 1, 0
        for i,v in enumerate(s):
            result += (ord(v) - 48) * (4 ** (final_index - i))
        return result

    def nb_sparse_fasta(self,filenames = [], klen = 10, max_size = -1 , merge_reverse_complement = True):
        ret = sps.vstack((sps.csr_matrix(_numba_array_fasta(np.memmap(name,  mode='r'), klen, max_size,merge_reverse_complement)) for  name  in filenames))
        ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
        return ret

    def nb_sparse_string(self, filenames = [], klen = 10, max_size = -1 , merge_reverse_complement = True):
        ret = sps.vstack((sps.csr_matrix(_numba_string(cont, klen, max_size, merge_reverse_complement)) for  cont  in filenames))
        ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
        return ret

    def nb_sparse_fastq(self, filenames = [], klen = 10, max_size = -1 , merge_reverse_complement = True):
        ret = sps.vstack((sps.csr_matrix(_numba_array_fastq(np.memmap(name,  mode='r'), klen, max_size, merge_reverse_complement)) for  name  in filenames))
        ret._meta = sps.eye(0, dtype = np.int16 ,format="csr")
        return ret
    
    def tfidf(X):
        return TfidfTransformer().fit_transform(X)

    def hellinger_dist_matrix( X):
        return pairwise_distances(np.sqrt(X), n_jobs=-1, metric='euclidean') / math.sqrt(2)
    
    @staticmethod
    @nb.njit(fastmath = True)
    def str_to_int(s):
        final_index, result = len(s) - 1, 0
        for i,v in enumerate(s):
            result += (ord(v) - 48) * (4 ** (final_index - i))
        return result
    
    def plot_clusters(data, algorithm, args, kwds):
        sns.set(rc={'figure.figsize':(10,10)}, style = "ticks")
        labels = algorithm(*args, **kwds).fit_predict(data)
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data.T[0], data.T[1], c=colors, **{'alpha' : 0.25, 's' : 80, 'linewidths':0})
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        #plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
        sns.despine()
    
