This repository contains a small example of how to apply jet energy
corrections (JEC), jet energy resolution smearing (JER) and the jet veto map to
NanoAOD samples.  The main demonstration is the ROOT macro
`applyJercAndJvmOnNano.C` which runs entirely in C++ using `correctionlib`.

Quick start from an lxplus environment:

* `ssh rverma@lxplus9.cern.ch`
* `cmsrel CMSSW_13_3_0/src`
* `cd CMSSW_13_3_0/src`
* `cmsenv`
* `git clone git@github.com:ravindkv/cms-jerc-application.git`
* `cd cms-jerc-application`
* `root -b -q applyJercAndJvmOnNano.C`
* `python applyJercAndJvmOnNano.py`
