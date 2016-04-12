import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as Math
import pylab as Plot
#from OriginalTSNE import tsne
from BHTSNEDropInReplacementTSNE import tsne 
import bloscpack as bp
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
	X = bp.unpack_ndarray_file("trainingpartition0.blp")
	labels = bp.unpack_ndarray_file("trainingpartition0.blp.labels")
	tmp = bp.unpack_ndarray_file("trainingpartition1.blp")
	tmpl = bp.unpack_ndarray_file("trainingpartition1.blp.labels")
	X = Math.concatenate((X,tmp))
	labels = Math.concatenate((labels,tmpl))
	tmp = bp.unpack_ndarray_file("trainingpartition2.blp")
	tmpl = bp.unpack_ndarray_file("trainingpartition2.blp.labels")
	X = Math.concatenate((X,tmp))
	labels = Math.concatenate((labels,tmpl))
	
	Y = tsne(X, 2, len(X[0]), 20.0);
	print(Y.shape)
	print(labels.shape)
	df = pd.DataFrame(Y, columns=['x', 'y'])
	df['label']=labels
	class1 = df.query("label == 1.0")
	class2 = df.query("label == 0.0")
	sns.set(style="darkgrid")	
	f, ax = plt.subplots(figsize=(8, 8))
	ax.set_aspect("equal")
	ax = sns.kdeplot(class2.x, class2.y,cmap="Greens", alpha=1,shade=True, shade_lowest=False)
	ax = sns.kdeplot(class1.x, class1.y,cmap="Reds", alpha=0.4,shade=True, shade_lowest=False)
	red = sns.color_palette("Reds")[-2]
	blue = sns.color_palette("Blues")[-2]
	ax.text(2.5, 8.2, "Class2", size=16, color=blue)
	ax.text(3.8, 4.5, "Class1", size=16, color=red)
	sns.plt.show()
