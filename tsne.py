from matplotlib import use
use('Agg')
import numpy as Math
import matplotlib.pyplot as Plot
#from OriginalTSNE import tsne
from BHTSNEDropInReplacementTSNE import tsne 

if __name__ == "__main__":
	print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
	print("Running example on 2,500 MNIST digits...")
	X = Math.loadtxt("mnist2500_X.txt");
	labels = Math.loadtxt("mnist2500_labels.txt");
	Y = tsne(X, 2, len(X[0]), 20.0);
	Plot.scatter(Y[:,0], Y[:,1], 20, labels);
	Plot.savefig("tsne.png");

