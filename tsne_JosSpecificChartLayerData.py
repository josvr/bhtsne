import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as Math
import gzip
import pickle
import pylab as Plot
#from OriginalTSNE import tsne
from BHTSNEDropInReplacementTSNE import processResultFileName
import bloscpack as bp
import pandas as pd
import seaborn as sns
import glob

if __name__ == "__main__":
  outputLabels = '/tmp/data/testpartition*.blp.labels'
  y = sorted(glob.glob(outputLabels))
  print("Found labels "+str(y))
  files = ['resultLayer7Perplexity20.000000.dat','resultLayer7Perplexity30.000000.dat','resultLayer7Perplexity5.000000.dat','resultLayer7Perplexity50.000000.dat']
  for f in files: 
        inputData =  f
        labels = bp.unpack_ndarray_file(y[0])
        for i in range(1,len(y)):
            tmpl = bp.unpack_ndarray_file(y[i])
            labels = Math.concatenate((labels,tmpl))
        Y = processResultFileName(inputData)
        if Y.shape[0] != labels.shape[0]:
            raise ValueError("X shape does not match label shape!!")    
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
        #ax.text(2.5, 8.2, "Class2", size=16, color=blue)
        #ax.text(3.8, 4.5, "Class1", size=16, color=red)
        sns.plt.savefig(inputData+".png")
