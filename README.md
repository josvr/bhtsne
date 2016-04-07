For my (Parttime) Master CS thesis project about deep learning I forked BHTSNE to make changes.

This Fork adds:

- (In my opinion) better Python wrapper.

  - NumpyArrayToDATADAT.py -> Convert a dumped Numpy Array to DATA.DAT
  - The C executable will then read DATA.DAT and generate RESULT.DAT
  - RESULTDATToNumpyArray.py -> Convert the result to a Numpy array
  - PlotOutput.py create a plot based on this and a labels array.

- Changes in C-code: parameters are printed to stdout
