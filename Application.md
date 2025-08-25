# cms-jerc-application

This package demonstrates how the JES, JER and jet veto map (JVM) provided by the JME-JERC subgroup can be applied on NanoAOD or MiniAOD samples.  It is designed so that analysts can copy the supplied configuration files, integrate the correction routines into their own code and cross‑check their outputs against the reference implementation: https://github.com/ravindkv/cms-jerc-application

## Setup the package

From a CMSSW environment (only needed for loading correctionlib):

```bash
ssh rverma@lxplus9.cern.ch
cmsrel CMSSW_13_3_0 && cd CMSSW_13_3_0/src && cmsenv
git clone git@github.com:ravindkv/cms-jerc-application.git
cd cms-jerc-application
```

## Apply on NanoAOD
```bash
cd ApplyOnNanoAOD 
root -b -q applyJercAndJvm.C
# or
python applyJercAndJvm.py
```

Inspect the printed log and the produced `output.root` file to verify the corrections.  Analysts may use the supplied configuration and functions as templates for their own workflows, skipping the AK4 or AK8 sections as needed.

### What this application provides

1. **Central configuration JSONs**  
   The files [`JercFileAndTagNamesAK4.json`](ApplyOnNanoAOD/JercFileAndTagNamesAK4.json), [`JercFileAndTagNamesAK8.json`](ApplyOnNanoAOD/jercFileAndTagNamesAK8.json) and [`JvmFileAndTagNames.json`](ApplyOnNanoAOD/JvmFileAndTagNames.json) hold the official tag names for the JES, JER and JVM payloads.  Users no longer need to assemble these strings themselves; future updates are the responsibility of the JERC conveners.
2. **Reference correction functions**  
   [`applyJercAndJvm.C`](ApplyOnNanoAOD/applyJercAndJvm.C) contains C++ implementations using `correctionlib`.  The accompanying [`applyJercAndJvm.py`](ApplyOnNanoAOD/applyJercAndJvm.py) shows the same logic in Python.  Analysts can copy or adapt these routines for their analyses.
3. **Validation against an example output**  
   After integration, results can be compared to the histograms produced by the reference macro to ensure a faithful implementation.
4. **Support for both AK4 and AK8 jets**  
   The package treats AK4 and AK8 jets in a unified way.  Analyses that only require one algorithm may simply ignore the other section.

## Apply on MiniAOD
To be implemented

### What this application provides
To be implemented


## Order of corrections

The correction sequence inside the provided macro follows the order used by the JERC group:

1. **Jet energy scale (JES)** – The raw jet four‑vectors are first corrected with the nominal JES factors.  For Monte Carlo samples, correlated JES uncertainty variations are then applied where both AK4 and AK8 share a common systematic source.
2. **Jet energy resolution (JER)** – After JES, jets in Monte Carlo samples are smeared to match the detector resolution.  Optional up/down variations are performed within regions defined in the JSON configuration.  
3. **Jet veto map (JVM)** – Once JES and JER have been applied, the jet veto map is queried to remove events containing jets in problematic detector regions.

Both AK4 and AK8 collections undergo the above steps.  The framework ensures that JES systematic shifts are applied in a correlated manner between the two algorithms so that analyses using both jet types maintain a consistent treatment. The name of the nuisance parameters obtained from up/down variations are named following the CAT's [naming convention](https://cms-analysis.docs.cern.ch/guidelines/uncertainty_digest/JME/).

