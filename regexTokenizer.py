import re
from itertools import chain
import numpy as np
import pandas as pd
import scipy.sparse as sps
import gzip
from collections import Counter

class regexKmerTokenizer():
    def __init__(self, klen = 30, ignoreNewLine = True, overlapping = True, requirements = -1, caller = 'tokenize_string', protein = False):
        self.protein = protein
        if self.protein:
            self.fastq_extractor = re.compile('\@[^\n]*\s+([a-zA-Z]+)\s+\+')
            self.fasta_extractor = re.compile('\>[^\n]*\s+([a-zA-Z\s]+)')
            self.fasta_extractor_complete = re.compile('\>([^\n]+)*\s+([a-zA-Z\s]+)')
            self.fastq_extractor_complete = re.compile('\@([^\n]+)*\s+([a-zA-Z]+)\s+\+[^\n]*\s+([^\s]+)\s+')
        else :
            self.fastq_extractor = re.compile('\@[^\n]*\s+([ATNCGatcg]+)\s+\+')
            self.fasta_extractor = re.compile('\>[^\n]*\s+([ATNCGatcg\s]+)')
            self.fasta_extractor_complete = re.compile('\>([^\n]+)*\s+([ATNCGatcg\s]+)')
            self.fastq_extractor_complete = re.compile('\@([^\n]+)*\s+([ATNCGatcg]+)\s+\+[^\n]*\s+([^\s]+)\s+')
        
        self.spacestrip = re.compile(r'\s*')
        self.Nstrip = re.compile('[N]+')
        self.klen = klen
        self.ignoreNewLine = ignoreNewLine
        self.overlapping = overlapping
        self.requirements = requirements
        self.caller = caller
        
        self.kmertok = self.kmerRegexMaker(klen, ignoreNewLine, overlapping, requirements)
        self.kmerfastatok = self.kmerRegexMaker(klen, True , overlapping, requirements)
        self.kmerfastqtok = self.kmerRegexMaker(klen, False , overlapping, requirements)
        
        
    def tokenize_fasta(self,fastastring, flatten = False):
        ret = [ self.kmerfastqtok.findall(self.spacestrip.sub('',string)) for string in self.fasta_extractor.findall(fastastring)]
        if flatten:
            return list(chain.from_iterable(ret))
        return ret
        
    def tokenize_fastq(self,fastqstring, flatten = False):
        ret = [self.kmerfastqtok.findall(string) for string in self.fastq_extractor.findall(fastqstring)]
        if flatten:
            return list(chain.from_iterable(ret))
        return ret
    
    def rm_spaces(self, string):
        return self.spacestrip.sub('', string.upper())
    
    def rm_N(self, string, addspace = False):
        convert2 = ' ' if addspace else ''
        return self.Nstrip.sub(convert2, string.upper())
    
    def rm_genomics_format(self, string, docformat ='fasta', flatten = False):
        if docformat == 'fasta':
            if flatten: return self.spacestrip.sub('', ''.join(self.fasta_extractor.findall(string)))
            return self.fasta_extractor.findall(string)
        if docformat == 'fastq':
            if flatten: return ''.join(self.fastq_extractor.findall(string))
            return self.fastq_extractor.findall(string)
        else: return ValueError('wrong format name')
        

    def tokenize_string(self, string):
        if self.ignoreNewLine:
            return self.kmerfastqtok.findall(self.spacestrip.sub('',string.upper()))
        return self.kmertok.findall(string.upper())
    
    def int2kmer(self,i):
        return np.base_repr(i,4).replace('0', 'A').replace('1', 'C').replace('2', 'G').replace('3', 'T').rjust(self.klen, 'A')
    
    def count_len(self, string):
        if self.protein:
            inter = re.sub('\[[\^a-zA-Z\s]+\]','?' ,string)
        else:
            inter = re.sub('\[[\^ATCGatcg\s]+\]','?' ,string)
        if '^' in inter: raise ValueError('for not statements, please do each character at a time. e.g. instead of "^AT" write [^A][^T]')
        return len(inter)

    def kmerRegexMaker(self, klen = 30, ignoreNewLine = True, overlapping = True, requirements = -1 ):
        
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
            a = requirementsadj[0]
            if a[0] == -1:
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
        
        
    def seq2pandas(self, text, columns = False, sep = False, docformat = 'fasta', from_file = False, tokenize = False, tokenAsInt = False,
                  countVectorizer = False, sample = -1, merge_reverse_complement = False):
        if from_file:
            if '.gz' in text:
                with gzip.open(text, 'rt') as file:
                    if docformat == 'fasta': 
                        df = pd.DataFrame( self.fasta_extractor_complete.findall(file.read()), columns = ['name', 'sequence'])
                        df.sequence = df.sequence.str.replace('\n', '')
                        df.sequence = df.sequence.str.replace('\r', '')
                        #df.sequence = df.sequence.apply(lambda x: self.spacestrip.sub('',x))
                    elif docformat == 'fastq':
                        df = pd.DataFrame(self.fastq_extractor_complete.findall(file.read()), columns = ['name', 'sequence', 'quality'])
            else:  
                with open(text) as file:
                    if docformat == 'fasta': 
                        df = pd.DataFrame( self.fasta_extractor_complete.findall(file.read()), columns = ['name', 'sequence'])
                        df.sequence = df.sequence.str.replace('\n', '')
                        df.sequence = df.sequence.str.replace('\r', '')
                        #df.sequence = df.sequence.apply(lambda x: self.spacestrip.sub('',x))
                    elif docformat == 'fastq':
                        df = pd.DataFrame(self.fastq_extractor_complete.findall(file.read()), columns = ['name', 'sequence', 'quality'])
                    
        elif docformat == 'fasta':
            df = pd.DataFrame(self.fasta_extractor_complete.findall(text), columns = ['name', 'sequence'])
            df.sequence = df.sequence.str.replace('\n', '')
            df.sequence = df.sequence.str.replace('\r', '')
            #df.sequence = df.sequence.apply(lambda x: self.spacestrip.sub('',x))
        elif docformat == 'fastq':
            df = pd.DataFrame(self.fastq_extractor_complete.findall(text), columns = ['name', 'sequence', 'quality'])
        else: return ValueError('wrong format name')
        
        if 0 < sample < 1 : df =df.sample(frac = sample)
        if sample >= 1: df =df.sample(sample)
        if (sep and columns):
            try:
                df[columns] = df.name.str.split(sep, expand= True)
                if 'name' not in columns:
                    df = df.drop('name', axis = 1)
                df = df.reindex(columns + ['sequence']+ ['quality'for x in range(1)  if docformat == 'fastq' ] , axis='columns')
            except:
                raise ValueError('missmatch between number of columns given and observed')
                
        ### limit numba use - problems with tensorflow
        if tokenAsInt or countVectorizer:
            from numba.typed import Dict
            from numba import types
            import numba as nb
            
            nbdict = Dict.empty(key_type=types.string,value_type=types.int64)
            nbdict['A'] = 0
            nbdict['C'] = 1
            nbdict['G'] = 2
            nbdict['T'] = 3
            
            @nb.njit(fastmath = True)
            def kmer2int(i, nbdict= nbdict):
                out =0
                for a in i:
                    out = out*4 + nbdict[a]
                return out

            @nb.njit(fastmath = True)
            def kmer2intReverseComp(i, nbdict= nbdict):
                out =0
                out2 = 0
                cnt = 0
                for a in i:
                    out = out*4 + nbdict[a]
                for a in i[::-1]:
                    out2 = out2*4 + (3-nbdict[a])
                return min(out, out2)


            def dict2sparse( dic):
                return sps.csr_matrix((list(dic.values()),[kmer2int(i, nbdict) for i in dic.keys()], [0, len(dic)]), shape = (1, 4**self.klen), dtype = np.int32)

            def dict2sparseRevComp( dic):
                return sps.csr_matrix((list(dic.values()),[kmer2intReverseComp(i, nbdict) for i in dic.keys()], [0, len(dic)]), 
                                      shape = (1, 4**self.klen - 4**(self.klen//2) )
                                     , dtype = np.int32)

            def dask_count_vec( seq = ''):
                ret = self.dict2sparse(Counter(self.kmertok.findall(seq)))
                ret._meta = sps.eye(0, format="csr", dtype =np.int32)
                return ret
                
            if countVectorizer:
                if not merge_reverse_complement: df['sparse_countvectorizer'] = df['sequence'].map(self.kmertok.findall).map(Counter).map(dict2sparse)
                if merge_reverse_complement: df['sparse_countvectorizer'] = df['sequence'].map(self.kmertok.findall).map(Counter).map(dict2sparseRevComp)
                return df
            
            if tokenAsInt:
                df['tokenized'] = df['sequence'].map(lambda x: [kmer2int(i, nbdict) for i in tok.kmertok.findall(x)])
                return df

        if tokenize:
            df['tokenized'] = df['sequence'].map(tok.kmertok.findall)

        return df
    
    
    
#if use_dask:
#    ddf = dd.from_pandas(df, npartitions=100)
#    if merge_reverse_complement: ddf['sparse_countvectorizer'] = ddf['sequence'].map(self.dask_count_vec, meta = (None, str))
#    if not merge_reverse_complement: ddf['sparse_countvectorizer'] = ddf['sequence'].map(self.dask_count_vec,meta = (None, str))
#    return ddf.compute(schedule='processes')

#if not use_dask:    
