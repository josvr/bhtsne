import numpy as Math
import pylab as Plot
#from OriginalTSNE import tsne
from BHTSNEDropInReplacementTSNE import tsne 
import bloscpack as bp

if __name__ == "__main__":
	X = bp.unpack_ndarray_file("trainingpartition0.blp")[0:15000,:]
	labels = bp.unpack_ndarray_file("trainingpartition0.blp.labels")[0:15000]
	print(X.shape)
	print(labels.shape)
	Y = tsne(X, 2, len(X[0]), 20.0);
	Plot.scatter(Y[:,0], Y[:,1], 20, labels,alpha=0.03);
	Plot.savefig("tsne.png");

