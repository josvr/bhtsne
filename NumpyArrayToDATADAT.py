#!/bin/python

from struct import pack
from os.path import join as path_join
import sys
import bloscpack as bp

input_matrix = '/home/josr/data/trainingpartition1.blp'
output_file = '/home/josr/data/data.dat'

theta = 0.5
perplexity = 50
no_dims = 2

# samples = pickle.load(input_matrix)
samples = bp.unpack_ndarray_file(input_matrix)

sample_dim = len(samples[0])
sample_count = len(samples)

print('Sample dimension: ',sample_dim)
print('Sample count: ',sample_count)

with open(path_join(output_file), 'wb') as data_file:
   data_file.write(pack('iiddi', sample_count, sample_dim, theta, perplexity, no_dims))
   for sample in samples:
     data_file.write(pack('{}d'.format(len(sample)), *sample))



