This repository contains a small example of how to apply jet energy corrections (JEC = JES correction + JER correction) and the jet veto map to NanoAOD and MiniAOD samples.  

Quick start from an lxplus environment:

* `ssh rverma@lxplus9.cern.ch`
* `cmsrel CMSSW_13_3_0/src`
* `cd CMSSW_13_3_0/src`
* `cmsenv`
* `git clone git@github.com:ravindkv/cms-jerc-application.git`
* `cd cms-jerc-application`

### Apply On NanoAOD

The main demonstration is the ROOT macro `applyJercAndJvm.C` which runs entirely in C++ using `correctionlib`. A python version is also provided.
* `cd ApplyOnNanoAOD`
* `root -b -q applyJercAndJvm.C`
or
* `python applyJercAndJvm.py`
Inspect the print log on the terminal and the output.root file

### Apply On MiniAOD
To be implemented
