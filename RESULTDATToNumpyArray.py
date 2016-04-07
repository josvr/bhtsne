from struct import unpack,calcsize
import bloscpack as bp
import numpy as np

inputFile = '/home/josr/data/result.dat'
outputFile = '/home/josr/data/result.blp'

def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

with open(inputFile, 'rb') as output_file:
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
r = np.asarray(r)
blosc_args=bp.BloscArgs(clevel=9)
bp.pack_ndarray_file(r,outputFile, chunk_size='100M', blosc_args=blosc_args)

