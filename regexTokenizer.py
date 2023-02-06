import re
from itertools import chain
import numpy as np
import pandas as pd
import scipy.sparse as sps
import gzip
from sklearn.metrics import pairwise_distances
from collections import Counter
import ahocorasick as ac
from typing import List, Set, Tuple, Dict
from IPython.display import display
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
import pickle



class regexKmerTokenizer():
    def __init__(self, klen = 15, ignoreNewLine = True, overlapping = True, \
                 requirements = -1, caller = 'tokenize_string', protein = False,\
                 use_dask = True):
        self.protein = protein
        if self.protein:
            self.fastq_extractor = re.compile('\@[^\n]*\s+([a-zA-Z]+)\s+\+').findall
            self.fasta_extractor = re.compile('\>[^\n]*\s+([a-zA-Z\s]+)').findall
            self.fasta_extractor_complete = re.compile('\>([^\n]+)*\s+([a-zA-Z\s]+)').findall
            self.fastq_extractor_complete = re.compile('\@([^\n]+)*\s+([a-zA-Z]+)\s+\+[^\n]*\s+([^\s]+)\s+').findall
        else :
            self.fastq_extractor = re.compile('\@[^\n]*\s+([ATNCGatcg]+)\s+\+').findall
            self.fasta_extractor = re.compile('\>[^\n]*\s+([ATNCGatcg\s]+)').findall
            self.fasta_extractor_complete = re.compile('\>([^\n]+)*\s+([ATNCGatncg\s]+)').findall
            self.fastq_extractor_complete = re.compile('\@([^\n]+)*\s+([ATNCGatncg]+)\s+\+[^\n]*\s+([^\s]+)\s+').findall
        
        self.klen = klen
        self.ignoreNewLine = ignoreNewLine
        self.overlapping = overlapping
        self.requirements = requirements
        self.caller = caller
        
        self.kmertok = self.kmerRegexMaker(klen, False, overlapping, requirements)
        self.getTokens = self.kmertok.findall
        self.tfw = str.maketrans({'A': '0', 'C': '1', 'G': '2', 'T': '3', 'a': '0', 'c': '1', 'g': '2', 't': '3'})
        self.trv = str.maketrans({'T': '0','G': '1','C': '2','A': '3', 't': '0', 'g': '1', 'c': '2', 'a': '3'})
        self.transInv = str.maketrans({'0': 'A', '1': 'C', '2': 'G', '3': 'T'})
        self.rmS = str.maketrans({'\n': '', '\t': '', '\r':'', ' ': ''})
        
        if use_dask:
            from dask.distributed import Client, client
            if not client._get_global_client(): 
                client = Client(processes = False)
                display(client)
            else: 
                display(client._get_global_client())
            
        
    def tokenize_fasta(self,fastastring: str, flatten = False)-> List[str]:#
        ret = [ self.getTokens(string.translate(self.rmS)) for string in self.fasta_extractor(fastastring)]
        if flatten:
            return list(chain.from_iterable(ret))
        return ret
        
    def tokenize_fastq(self,fastqstring: str, flatten = False)-> List[str]:#
        ret = [self.getTokens(string) for string in self.fastq_extractor(fastqstring)]
        if flatten:
            return list(chain.from_iterable(ret))
        return ret
    
    def rm_spaces(self, string: str)-> str:
        return string.translate(self.rmS)
    
    def rm_N(self, string: str, addspace = False)-> str:
        switch = ' ' if addspace else ''
        return string.translate(str.maketrans({'N': switch}))
    
    def rm_genomics_format(self, string, docformat ='fasta', flatten = False):
        if docformat == 'fasta':
            if flatten: return ''.join(self.fasta_extractor(string)).translate(self.rmS)
            return self.fasta_extractor(string).translate(self.rmS)
        if docformat == 'fastq':
            if flatten: return ''.join(self.fastq_extractor(string))
            return self.fastq_extractor(string)
        else: return ValueError('wrong format name')
        

    def tokenize_string(self, string: str)-> List[str]:
        if self.ignoreNewLine:
            return self.getTokens(string.upper().translate(self.rmS))
        return self.getTokens(string.upper())
    
    def int2kmer(self,i: int)-> str:
        return np.base_repr(i,4).translate(self.transInv).rjust(self.klen, 'A')
    
    def kmer2int(self,s: str) -> int:
        return int(s.translate(self.tfw), base = 4)

    def kmer2intRevComp(self, s: str)-> int:
        return min(int(s.translate(self.tfw), base = 4), int(s[::-1].translate(self.trv), base = 4))
    
    def count_len(self, string: str)-> int:
        if self.protein:
            inter = re.sub('\[[\^a-zA-Z\s]+\]','?' ,string)
        else:
            inter = re.sub('\[[\^ATCGatcg\s]+\]','?' ,string)
        if '^' in inter: raise ValueError('for not statements, please do each character at a time. e.g. instead of "^AT" write [^A][^T]')
        return len(inter)

    def kmerRegexMaker(self, klen = 15, ignoreNewLine = True, overlapping = True, requirements = -1 ):
        
        if self.protein:
            seq2add = '[a-zA-Z]' 
        else:
            seq2add = '[ATCG]' 
        ig =  '\s*' #'(?:\s*)?'
        if ignoreNewLine:
            seq2add += ig

        if requirements == -1:
            outstring = seq2add*klen
        else:
            outstring = ''
            req2 = [x if (x[0] >= 0 or x[0]==-1) else (klen+x[0], x[1]) for x in  requirements]
            requirementsadj = sorted(req2,key=lambda tup: tup[0])
            if (a := requirementsadj[0])[0] == -1:
                requirementsadj = requirementsadj[1:] + [(klen - self.count_len(a[1]), a[1])]

            cnt = 0
            for pos, req in requirementsadj:
                outstring += seq2add*(pos - cnt)
                if self.protein: reqadj = re.sub(r'(\[\^[a-zA-Z]+)', r'\1\\s+\\d+', req)
                else: reqadj = re.sub(r'(\[\^[ATCG]+)', r'\1\\s+\\d+', req)
                outstring += reqadj + ig if  ignoreNewLine  else reqadj
                cnt = self.count_len(req)+pos
            outstring += seq2add*(klen - cnt)

        if ignoreNewLine:
            outstring = outstring[:-len(ig)]

        if overlapping:
            outstring = r'(?=('+ outstring + '))'
        return re.compile(outstring, re.A) #re.A for ascii only
    
    def __call__(self, items,  **kwargs):
        if type(items)== str:
            return (getattr(self, self.caller)(items, **kwargs))
        return (getattr(self, self.caller)(i, **kwargs) for i in items)
        
    def txt2df(self, txt: str, fmt: str) -> pd.DataFrame:
        if fmt == 'fasta':
            return pd.DataFrame(self.fasta_extractor_complete(txt), columns = ['name', 'sequence'])
        if fmt == 'fastq':
            return pd.DataFrame(self.fastq_extractor_complete(txt), columns = ['name', 'sequence', 'quality'])
        raise ValueError('wrong format name')
        
    
    def dict2sparse(self, dic: dict)-> sps.csr_matrix:
        return sps.csr_matrix((list(dic.values()),[self.kmer2int(i) for i in dic.keys()], [0, len(dic)]), shape = (1, 4**self.klen), dtype = np.int32)

    def dict2sparseRevComp(self, dic: dict)-> sps.csr_matrix:
        return sps.csr_matrix((list(dic.values()),[self.kmer2intRevComp(i) for i in dic.keys()], [0, len(dic)]), 
                              shape = (1, 4**self.klen - 4**(self.klen//2) )
                             , dtype = np.int32)
    
    def seq2csr(self, seq: str) -> sps.csr_matrix:
        return self.dict2sparse(Counter(self.getTokens(seq)))
    
    def seq2csrRevComp(self, seq: str) -> sps.csr_matrix:
        return self.dict2sparseRevComp(Counter(self.getTokens(seq)))
    
    def sparseStack(self,seqlist = []) -> sps.csr_matrix:
        ret = sps.vstack(self.seq2csr(seq) for  seq  in seqlist)
        ret._meta = sps.eye(0, dtype = np.uint32 ,format="csr")
        return ret
    
    def sparseStackRevComp(self,seqlist = []) -> sps.csr_matrix:
        ret = sps.vstack(self.seq2csrRevComp(seq) for  seq  in seqlist)
        ret._meta = sps.eye(0, dtype = np.uint32 ,format="csr")
        return ret
    
    def addKmerAutomaton(self,A, seq:str, info: Tuple[str], rmDuplicates = False, subsample: int = 1)->bool:
        for kmer in self.getTokens(seq)[::subsample]:
            if rmDuplicates: 
                if not A.add_word(kmer, info): A.pop(kmer)
            else:
                if kmer in A: A.add_word(kmer, A.pop(kmer) | set([info]))
                else: A.add_word(kmer, set([info]))
        return True
        
    def addSequenceAutomaton(self,A, seq:str,info: Tuple[str])->bool:
        if not A.add_word(seq, info): A.pop(kmer) 
        return True
    
    def daskaddseqAutomaton(self, df: pd.DataFrame, A, columns2add: List[str], rmDuplicates: bool = False,subsample: int = 1 ):
        df.apply(lambda x: self.addSequenceAutomaton(A, x.sequence, tuple(x[columns2add]), rm_spaces, subsample),axis = 1)
        return True
    
    def daskaddkmerAutomaton(self, df: pd.DataFrame, A, columns2add: List[str], rmDuplicates: bool = False):
        df.apply(lambda x: self.addKmerAutomaton(A, x.sequence, tuple(x[columns2add])),axis = 1)
        return True
         
    def makeAutomaton(self, df: pd.DataFrame, mode = 'pandas', kmerize = True, savename = False,\
                     columns2add: List[str] = ['name'], rmDuplicates: bool = False, slidingWindow: int = 1):
        A = ac.Automaton()
        if mode=='dask':
            import dask.dataframe as dd
            ddf = dd.from_pandas(df.reset_index(), npartitions=100) 
            if kmerize:
                _ = ddf.map_partitions(self.daskaddkmerAutomaton, A = A,columns2add= columns2add,\
                                       rmDuplicates= rmDuplicates, subsample= slidingWindow, meta = list).compute()
            else:
                _ = ddf.map_partitions(self.daskaddseqAutomaton,A = A,columns2add= columns2add,\
                                       rmDuplicates= rmDuplicates,subsample=slidingWindow ,meta = list).compute()

        if mode == 'pandas':
            if kmerize:
                df.apply(lambda x: self.addKmerAutomaton(A, x.sequence, tuple(x[columns2add]), rmDuplicates, slidingWindow), axis = 1)
            else:
                df.apply(lambda x: self.addSequenceAutomaton(A, x.sequence, tuple(x[columns2add]),rmDuplicates, slidingWindow), axis = 1)
            
        A.make_automaton()
        display(A.get_stats())
        if savename:
            A.save(savename, pickle.dumps)
        return A    
    
    def loadAutomaton(filename: str):
        return ac.load(filename, pickle.loads)
    
    
    def mapRead(self, seq: str, A, percentMatch: float = 0, multipleMappingLimit: int = np.inf, key_len: int = 1) -> List[str]:
        ABC = Counter(chain.from_iterable(x[1] for x in A.iter(seq)))
        if not ABC: return ('!noMatch',)*key_len
        cmlis = ABC.most_common()   
        mc, mcf = cmlis[0]
        if len(cmlis)>1:
            if cmlis[0][1] == cmlis[1][1]:
                return [f'!mm:{cmlis[0][0][0]}|{cmlis[1][0][0]}']*key_len
        if (pm := (mcf + self.klen - 1)/len(seq)) < percentMatch:
            return [f'!match%<{pm}']*key_len
        if (mm := sum(ABC.values()) - mcf + self.klen -1)/len(seq) > multipleMappingLimit:
            return [f'!mm>{mm}']*key_len
        if type(mc) == str:
            return [mc]
        return mc
    
    def dfseq2dfgene(self, df: pd.DataFrame, A: ac.Automaton, mode: str = 'dask',\
                     percentMatch: float = 0, multipleMappingLimit: int = np.inf, columnsNames = ['mappedRead']) -> pd.DataFrame:
        if type(columnsNames) == str: columnsNames = [columnsNames]
        key_len = len(columnsNames)
        if mode == 'pandas':
            df[columnsNames] = df.sequence.map(lambda x: self.mapRead(x, A, percentMatch, multipleMappingLimit, key_len)).tolist() #
        elif mode == 'dask':
            import dask.dataframe as dd
            ddf = dd.from_pandas(df.reset_index(), npartitions=100) 
            df[columnsNames] = ddf['sequence'].apply(self.mapRead, A = A, \
                                                            percentMatch = percentMatch, \
                                                            multipleMappingLimit = multipleMappingLimit,\
                                                            key_len = key_len ,meta = list).compute().to_list()
        return df
            
    
    def seq2pandas(self, text : str, columns: List[str] = False, sep = False, \
                   docformat: str = 'fasta', from_file = False, tokenize: bool = False,\
                   tokenAsInt: bool = False,countVectorizer: bool = False, sample = -1,\
                   merge_reverse_complement: bool = False, mode: str = 'dask', \
                   random_state: int = -1) -> pd.DataFrame:
        
        if from_file:
            if '.gz' in text:
                with gzip.open(text, 'rt') as file:
                    df = self.txt2df(file.read(), docformat)
            else:  
                with open(text) as file:
                    df = self.txt2df(file.read(), docformat)                 
        else:
            df = self.txt2df(text, docformat)
        if docformat == 'fasta':
            df.sequence = df.sequence.str.replace('\n|\r', '', regex = True)
        
        if random_state == -1 and sample != -1: 
            random_state = np.random.choice(range(100))
            print(f'random_state set to {random_state}')
        if 0 < sample < 1 : df =df.sample(frac = sample,random_state = random_state)
        if sample >= 1: df =df.sample(sample,random_state = random_state)
        if (sep and columns):
            try:
                df[columns] = df.name.str.split(sep, expand= True)
                if 'name' not in columns:
                    df = df.drop('name', axis = 1)
                df = df.reindex(columns + ['sequence']+ ['quality' for x in range(1)  if docformat == 'fastq'] , axis='columns')
            except:
                raise ValueError('missmatch between number of columns given and observed')
                            
        if mode == 'dask':
            import dask.dataframe as dd
            ddf = dd.from_pandas(df.reset_index(), npartitions=120) 
            
            if countVectorizer:
                if not merge_reverse_complement:
                    #df[['sparse_countvectorizer']] = ddf['sequence'].map_partitions(self.sparseStack).compute() #,meta = ddf
                    df['sparse_countvectorizer'] = ddf['sequence'].map_partitions(np.vectorize(self.seq2csr)).compute()
                if merge_reverse_complement:
                    #df[['sparse_countvectorizer']] = ddf['sequence'].map_partitions(self.sparseStackRevComp).compute()
                    df['sparse_countvectorizer'] = ddf['sequence'].map_partitions(np.vectorize(self.seq2csrRevComp)).compute()
                return df
            if tokenAsInt:
                df['tokenized'] = ddf['sequence'].apply(lambda x: [self.kmer2int(i) for i in self.getTokens(x)], meta = ddf).compute()
                return df
            if tokenize:
                df['tokenized'] = ddf['sequence'].apply(self.getTokens, meta = ddf).compute()
                

        elif mode == 'pandas':
            if countVectorizer :
                if not merge_reverse_complement: df['sparse_countvectorizer'] = df['sequence'].map(self.getTokens).map(Counter).map(self.dict2sparse)
                if merge_reverse_complement: df['sparse_countvectorizer'] = df['sequence'].map(self.getTokens).map(Counter).map(self.dict2sparseRevComp)
                return df

            if tokenAsInt:
                df['tokenized'] = df['sequence'].map(lambda x: [kmer2int(i, nbdict) for i in self.getTokens(x)])
                return df
            if tokenize:
                df['tokenized'] = df['sequence'].map(self.getTokens)
                
        elif mode == 'vaex':
            import vaex
            vdf = vaex.from_pandas(df)
            if countVectorizer:
                if not merge_reverse_complement:
                    vdf['sparse_countvectorizer'] = vdf['sequence'].apply(self.seq2csr)
                if merge_reverse_complement:
                    vdf['sparse_countvectorizer'] = vdf['sequence'].apply(self.seq2csrRevComp)
                return vdf.to_pandas_df().set_index(df.index)
            if tokenAsInt:
                vdf['tokenized'] = vdf['sequence'].apply(lambda x: [self.kmer2int(i) for i in self.getTokens(x)])
                return vdf.to_pandas_df().set_index(df.index)
            if tokenize:
                vdf['tokenized'] = vdf['sequence'].apply(self.getTokens)
                df = vdf.to_pandas_df().set_index(df.index)
            
        return df
    
    def sparseStackFasta(self, filenames = [],  revComp = False) -> sps.csr_matrix:
        if revComp: ret = sps.vstack(self.dict2sparseRevComp(Counter(self.tokenize_fasta('>\n'+ open(file).read(), flatten = True))) for  file  in filenames)
        else: ret = sps.vstack(self.dict2sparse(Counter(self.tokenize_fasta('>\n' + open(file).read(), flatten = True))) for  file  in filenames)
        ret._meta = sps.eye(0, dtype = np.uint32 ,format="csr")
        return ret        
    
    def sparseStackFastq(self, filenames = [],  revComp = False) -> sps.csr_matrix:
        if revComp: ret = sps.vstack(self.dict2sparseRevComp(Counter(self.tokenize_fastq( open(file).read(), flatten = True))) for  file  in filenames)
        else: ret = sps.vstack(self.dict2sparse(Counter(self.tokenize_fastq(open(file).read(), flatten = True))) for  file  in filenames)
        ret._meta = sps.eye(0, dtype = np.uint32 ,format="csr")
        return ret     
                
    def data2csr(self, files = [], tfidf = False, binary_output = False, \
                 content = 'auto', single_row_output = False, minorallelefrequency = 0, MergeReverseComplement = False):
        self.revComp = MergeReverseComplement
        self.n_docs = 0
        
        ncols = 4**self.klen - 4**(self.klen//2) if self.revComp else 4**self.klen
        if content == 'auto':
            if (typ := files[0].split('.')[-1]) == files[0]: typ = 'string'
        else: typ = content
            
        import dask.bag as db
        import dask.array as da
        b = db.from_sequence(files)
        if typ == 'fastq':
            bag2 = b.map_partitions(self.sparseStackFastq,  revComp = self.revComp)
        elif typ == 'fasta' or typ == 'fa':
            bag2 = b.map_partitions(self.sparseStackFasta,  revComp = self.revComp)
        elif typ == 'string':
            if self.revComp:
                bag2 = b.map_partitions(self.sparseStackRevComp)
            else:
                bag2 = b.map_partitions(self.sparseStack)
        else: return 'content type not found'

        objs = bag2.to_delayed()
        arrs = [da.from_delayed(obj, (np.nan, ncols), str) for obj in objs]
        ret = da.concatenate(arrs, axis=0).compute()
        
        self.n_docs = ret.shape[0]
        
        if minorallelefrequency:
            if type(minorallelefrequency) == int or type(minorallelefrequency) == float:
                self.maf = sorted([minorallelefrequency, 1-minorallelefrequency])
            elif type(minorallelefrequency) == list or type(minorallelefrequency) == tuple:
                self.maf = sorted([minorallelefrequency[0], minorallelefrequency[1]])
            else:
                raise ValueError('Minor Allele Fequency not accepted')
        else: self.maf =0
                
        if self.maf:
            self.n_docs = ret.shape[0]
            idx = np.unique(ret.indices, return_counts=True)
            self.acceptedindices  =  idx[0][(self.maf[1]*self.n_docs >= idx[1])&(idx[1]>= self.maf[0]*self.n_docs)]
            if len(self.acceptedindices) == 0:  raise ValueError('No kmers present with minor allele frequency range sent')
            ret =  ret[:,  self.acceptedindices ]
        
        else:
            self.acceptedindices = np.unique(ret.indices)
        
        if tfidf: ret =  TfidfTransformer().fit_transform(ret)
        if binary_output: ret.data.fill(1)
        if single_row_output: 
            ret = sps.csr_matrix((ret.data, ret.indices, (0, ret.indptr[-1])), dtype = np.int16, copy=False, shape = (1, ncols))
            self.n_docs = 1
        del arrs, objs, bag2, b
        return ret
    
    def applymaf(self, X, maf, reset_maf = True):
        if type(maf) == int or type(maf) == float:mafed = sorted([maf, 1-maf])
        elif type(maf) == list or type(maf) == tuple: mafed = sorted([maf[0], maf[1]])
        else: raise ValueError('Minor Allele Fequency not accepted')
        if reset_maf:
            warnings.warn(f"maf of Class {self} was set to {mafed}, indices are stored in self.acceptedIndices")
        else:
            warnings.warn(f"maf of Class {self} was not reset")
        n_doc = X.shape[0]
        idx = np.unique(X.indices, return_counts=True)
        out = idx[0][(mafed[1]*n_doc >= idx[1])&(idx[1]>= mafed[0]*n_doc)]
        #self.acceptedindices  =  idx[0][(self.maf[1]*self.n_docs >= idx[1])&(idx[1]>= self.maf[0]*self.n_docs)]
        if reset_maf:    
            self.maf = mafed
            self.acceptedindices = out
        return X[:, out] #X.tolil()[:,out].tocsr()
    
    def removeEmptyColumns(self, X):
        return X[:np.unique(X.indices)]#X.tolil()[:, np.unique(X.indices)].tocsr()
    
    def reduceKmerMatrix(self,X):
        warnings.warn(f"using indices stored in self.acceptedindices") 
        return X[:,self.acceptedindices]#X.tolil()[:,self.acceptedindices].tocsr()
    
    def filterLowCount(self, X, minVal):
        A = X.copy()
        A.data *= A.data>= minVal
        A.eliminate_zeros()
        return A
    
    def removeMultipleMatchKmers(self,X, reset_maf = True):
        warnings.warn(f"indices will be stored in self.acceptedindices") 
        n_doc = X.shape[0]
        idx = np.unique(X.indices, return_counts=True)
        out = idx[0][ idx[1] == 1]
        if reset_maf:
            self.acceptedindices  =  out
            self.maf = (0, 1/n_doc)
        return X[:, out]#X.tolil()[:, out].tocsr()
    
    def normalizeGeneSize(self,X):
        return sps.csr_matrix(X/X.sum(axis = 1))

    def dic2sparserow(dic, maxsize =  int(1e8)):
        return sps.csr_matrix((list(dic.values()), list(dic.keys()), [0, len(dic)]), shape = (1, maxsize ))        
    

    def kmer_list(self, matrix = None):
        if matrix != None:
            return [self.int2kmer(i) for i in self.acceptedindices]
        else:     
            return [self.int2kmer(i) for i in self.acceptedindices]
        
    
    def tfidf(X):
        return TfidfTransformer().fit_transform(X)

    def hellinger_dist_matrix(X):
        return pairwise_distances(np.sqrt(X), n_jobs=-1, metric='euclidean') / math.sqrt(2)
    
    def plot_clusters(data, algorithm, args, kwds):
        labels = algorithm(*args, **kwds).fit_predict(data)
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data.T[0], data.T[1], c=colors, **{'alpha' : 0.25, 's' : 80, 'linewidths':0})
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        sns.despine()
    

    
