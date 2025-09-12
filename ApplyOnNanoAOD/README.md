
### Apply On NanoAOD

The main demonstration is the ROOT macro `applyJercAndJvm.C` which runs entirely in C++ using `correctionlib`. A python version is also provided.
* `root -b -q applyJercAndJvm.C`
or
* `python applyJercAndJvm.py`

Inspect the print log on the terminal and the output.root file

To visualize Data/MC comparisons with uncertainty bands across all eras,
run the plotting helper:

```
python plotOutput.py 
```

This scans the ROOT file for all years and eras defined in the config and
writes a single multi-page PDF (`output_hJetPt_AK4.pdf`) in the current
directory.

