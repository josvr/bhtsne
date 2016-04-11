For my (Parttime) Master CS thesis project about deep learning I forked BHTSNE to make changes.

**This Fork adds:**

- (In my opinion) better Python wrapper. It is a 'drop in replacement' following the original tsne interface specifications.

**What it does:**

`tsne.py` reads MNIST(subset) and plot a figure.

Switch between Original TSNE and BH-SNE by commenting out `OriginalTSNE` or `BHTSNEDropInReplacement` in `tsne.py`:

```python
#from OriginalTSNE import tsne
from BHTSNEDropInReplacementTSNE import tsne
```

I removed the PCA dim. reduction from the original TSNE. 

- Changes in C-code: parameters are printed to stdout

TODO: Generated intermediate files are not being removed. 
