import numpy as np   
import scipy.sparse as sp

from libc.stdlib cimport malloc, free
from libc.stdio cimport fopen, fclose, FILE, EOF, fgetc, feof
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cdef int char2int(char chars) nogil:
    if chars == 65: return 0
    elif chars == 67: return 1
    elif chars == 71: return 2
    elif chars == 84: return 3
    return -1

cdef int invchar2int(char chars) nogil:
    if chars == 65: return 3
    elif chars == 67: return 2
    elif chars == 71: return 1
    elif chars == 84: return 0
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cdef cykmerarray_fastq(str filename, int klen, unsigned long max_size = -1, int mg_fw_rv = 1): ##unsigned long is up to 32bp cnp.uint16_t[:]
    """Efficiently read in a file"""
    cdef FILE *fp = NULL # create a file pointer
    fp = fopen(filename.encode(encoding='utf-8'), "rb")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
    # file parsing variables
    cdef unsigned long array_size = 4**klen 
    if mg_fw_rv : array_size -= 4**(klen//2)
    cdef unsigned long fw_div = 4**(klen-1)
    
    if max_size > 0 and max_size < array_size:
        array_size = max_size
    
    cdef char c 
    cdef int i
    cdef int is_line = 1 
    cdef int window_pos = 0
    cdef bytes pywindow 
    cdef unsigned long fw_num = 0
    cdef unsigned long rv_num = 0
    #cdef list col = []
    cdef cnp.ndarray[cnp.int16_t, ndim=1] result = np.zeros((array_size), dtype = np.int16)
    
    with nogil:
        if mg_fw_rv:
            # maybe try while (c := fgetc(fp)) 
            while 1 :
                c = fgetc(fp)
                if feof(fp): break
                if c == 78: window_pos = 0
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
            while 1 :
                c = fgetc(fp)
                if feof(fp): break
                if c == 78: window_pos = 0
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
    # close the file
    fclose(fp)
    return result

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cdef cykmerarray_fasta(str filename, int klen, unsigned long max_size = -1, int mg_fw_rv = 1): ##unsigned long is up to 32bp cnp.uint16_t[:]
    """Efficiently read in a file"""
    cdef FILE *fp = NULL # create a file pointer
    fp = fopen(filename.encode(encoding='utf-8'), "rb")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file or directory: '%s'" % filename)
    # file parsing variables
    cdef unsigned long array_size = 4**klen
    if mg_fw_rv: array_size -= 4**(klen//2)
    cdef unsigned long fw_div = 4**(klen-1)
    
    if max_size > 0 and max_size < array_size:
        array_size = max_size
    
    cdef char c 
    cdef int i
    cdef int is_line = 1 
    cdef int window_pos = 0
    cdef bytes pywindow 
    cdef unsigned long fw_num = 0
    cdef unsigned long rv_num = 0
    cdef cnp.ndarray[cnp.int16_t, ndim=1] result = np.zeros((array_size), dtype = np.int16)

    with nogil:
        if mg_fw_rv:
            while 1 :
                c = fgetc(fp)
                if feof(fp): break
                if c == 78: window_pos = 0
                elif c == 10 or c==13: pass
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
            while 1 :
                c = fgetc(fp)
                if feof(fp): break
                if c == 78: window_pos = 0
                elif c == 10 or c == 13:  pass
                else: 
                    fw_num = fw_num*4 + char2int(c)
                    window_pos += 1
                    if window_pos == klen: 
                        result[fw_num%array_size] += 1
                        fw_num %= fw_div
                        window_pos -= 1
    # close the file
    fclose(fp)
    return result


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)
#cdef cykmerarray_string(char* string, int klen, unsigned long max_size = -1, int mg_fw_rv = 1): ##unsigned long is up to 32bp cnp.uint16_t[:] char *string
#    
#    cdef unsigned long array_size = 4**klen 
#    if mg_fw_rv : array_size -= 4**(klen//2)
#    cdef unsigned long fw_div = 4**(klen-1)
#    
#    if max_size > 0 and max_size < array_size:
#        array_size = max_size
#    cdef char c 
#    cdef int i
#    cdef int is_line = 1 
#    cdef int window_pos = 0
#    cdef bytes pywindow 
#    cdef unsigned long fw_num = 0
#    cdef unsigned long rv_num = 0
#    cdef cnp.ndarray[cnp.int16_t, ndim=1] result = np.zeros((array_size), dtype = np.int16)
#    
#    if mg_fw_rv:
#        for c in string:
#            if c == 78: window_pos = 0
#            elif c == 10: pass
#            else: 
#                fw_num = fw_num*4 + char2int(c)
#                rv_num = invchar2int(c)*(4**window_pos) + rv_num
#                window_pos += 1
#                if window_pos == klen: 
#                    if fw_num < rv_num: result[fw_num%array_size] += 1
#                    else: result[rv_num%array_size] += 1
#                    fw_num %= fw_div
#                    rv_num //= 4
#                    window_pos -= 1
#    else:
#        for c in string:
#            if c == 78: window_pos = 0
#            elif c == 10:  pass
#            else: 
#                fw_num = fw_num*4 + char2int(c)
#                window_pos += 1
#                if window_pos == klen: 
#                    result[fw_num%array_size] += 1
#                    fw_num %= fw_div
#                    window_pos -= 1
#    return result

    
def cyfiles2sparse(filenames = [], int klen = 13, max_size = -1, merge_reverse_complement = 1):
    
    ##check if files are the same extension
    #if sum([x.split('.')[-1] == filenames[0].split('.')[-1] for x in filenames]) != len(filenames):
    #    print('there are files of different types!')
    #    return -1
    
    extension = filenames[0].split('.')[-1]
    if extension == filenames[0]: extension = 'string'
        
    cdef long size = 4**klen 
    if merge_reverse_complement: size -= 4**(klen//2)  #cdef long long 
    if max_size > 0 and max_size < size:
        size = max_size
    cdef int corpus_size = len(filenames)
    
    if extension == 'fastq':
        ret = sp.vstack((sp.csr_matrix(cykmerarray_fastq(name, klen, size, merge_reverse_complement)) for  name  in filenames))
        ret._meta = sp.eye(0, dtype = np.int16 ,format="csr")
        
    elif extension == 'fasta':
        ret = sp.vstack((sp.csr_matrix(cykmerarray_fasta(name, klen, size, merge_reverse_complement)) for  name  in filenames))
        ret._meta = sp.eye(0, dtype = np.int16 ,format="csr")
    
#    elif extension == 'string':
#        ret = sp.vstack((sp.csr_matrix(cykmerarray_fasta(name.encode(encoding='ascii'), klen, size, merge_reverse_complement)) for  name  in filenames))
#        ret._meta = sp.eye(0, dtype = np.int16 ,format="csr")       
    else:
        raise ValueError('unknown file type')
        return -1
    
    return ret