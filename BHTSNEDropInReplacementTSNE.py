import numpy as Math
from struct import pack,unpack,calcsize
from os.path import join as path_join
import sys
from subprocess import call

def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
  createDataFile(X,no_dims,initial_dims,perplexity)
  callBHTSNE()
  return processResultFile()

def createDataFile(X,no_dims,initial_dims,perplexity):
  theta = 0.5
  sample_count = len(X)
  with open('data.dat', 'wb') as data_file:
     data_file.write(pack('iiddi', sample_count, initial_dims, theta, perplexity, no_dims))
     for sample in X:
       data_file.write(pack('{}d'.format(len(sample)), *sample))

def callBHTSNE():
  call(["./bh_tsne"])    

def processResultFile(): 
  with open('result.dat', 'rb') as output_file:
    # The first two integers are just the number of samples and the
    #   dimensionality
    result_samples, result_dims = _read_unpack('ii', output_file)
    # Collect the results, but they may be out of order
    results = [_read_unpack('{}d'.format(result_dims), output_file)
        for _ in range(0,result_samples)]
    # Now collect the landmark data so that we can return the data in
    #   the order it arrived
    results = [(_read_unpack('i', output_file), e) for e in results]
    # Put the results in order and yield it
    results.sort()
  # strip of the landmark
  r = [ x[1] for x in results ]
  r = Math.asarray(r)
  return r;

def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


 
