import bloscpack as bp
import numpy as Math
import pylab as Plot

labelsFile = '/home/josr/data/trainingpartition1.blp.labels'
tsneMatrixFile = '/home/josr/data/result.blp'
plotFile = "/home/josr/data/plot.png"

labels=bp.unpack_ndarray_file(labelsFile).view(dtype=Math.int32)
Y= bp.unpack_ndarray_file(tsneMatrixFile);
print(labels.shape)
print(Y[:,0])
Plot.scatter(Y[:,0], Y[:,1], 20, labels);
Plot.savefig(plotFile)
