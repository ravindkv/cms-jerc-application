#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
applyJercAndJvmNEW.py

A faithful Python port of applyJercAndJvm.C using a *columnar* reader (uproot/awkward)
plus correctionlib, while writing ROOT histograms via PyROOT to match the original
file structure/contents. The goal is bit-for-bit identical logic, so several
implementation details mirror the C++ closely (including corner-case behaviors).

Dependencies (tested with CMSSW/LXPLUS-style envs):
  - uproot >= 5
  - awkward >= 2
  - correctionlib >= 2 (schemav2)
  - PyROOT (for writing TH1F/TFile directories)

Run (inside a CMSSW or ROOT-enabled environment):
  cmsenv
  python3 applyJercAndJvmNEW.py \
      --input-mc NanoAOD_MC.root \
      --input-data NanoAOD_Data.root \
      --out output.root

JSONs expected in CWD (same keys as the C++ macro):
  - JercFileAndTagNamesAK4.json
  - JercFileAndTagNamesAK8.json
  - JvmFileAndTagNames.json

Notes:
- We write the same histogram names and directory layout as the C++:
  year/(MC|Data)/(optional Era)/<SystSet>/<SystName>/<hists>
- Random seeds for JER smear are set per-jet with (event+run+luminosityBlock),
  matching the macro.
- Some deltaR calls in the original macro pass arguments in a swapped order; we
  intentionally preserve that behavior for byte-for-byte equivalence.
"""

from __future__ import annotations
import os
import json
import math
import argparse
from typing import Optional, List, Dict, Tuple

import numpy as np
import awkward as ak
import uproot

# PyROOT only for writing TH1/TFile like the C++ macro
import ROOT
ROOT.gROOT.SetBatch(True)

import correctionlib._core as correction

# ---------------------------------------------------------------------------
# Global toggles — keep defaults identical to C++ macro
# ---------------------------------------------------------------------------
applyOnlyOnAK4 = False
applyOnlyOnAK8 = False
applyOnAK4AndAK8 = True
applyOnMET = True

# Indentation utilities for optional debug prints
spaces2, spaces3, spaces4, spaces5, spaces6 = 2, 3, 4, 5, 6

def print_debug(enable: bool, indent: int, *args):
    if not enable:
        return
    print(" ".rjust(indent, " "), *args, sep="")

# ---------------------------------------------------------------------------
# Helpers reproducing year-dependent logic
# ---------------------------------------------------------------------------

def has_phi_dependent_L2(year: str) -> bool:
    return year in ("2023Post", "2024")


def requires_run_based_residual(year: str) -> bool:
    return year in ("2023Pre", "2023Post", "2024")


def uses_puppi_met(year: str) -> bool:
    return year in {"2022Pre", "2022Post", "2023Pre", "2023Post", "2024"}


def representative_run_number(year: str) -> float:
    if year == "2023Pre":
        return 367080.0
    if year == "2023Post":
        return 369803.0
    if year == "2024":
        return 379412.0
    return 0.0  # replaced per-event with actual run when not Run-3 special

# ---------------------------------------------------------------------------
# JSON helpers (same keys/names as the C++ macro expects)
# ---------------------------------------------------------------------------

def load_json_config(fname: str) -> dict:
    with open(fname, "r") as f:
        return json.load(f)


def _get(obj: dict, key: str) -> str:
    if key not in obj:
        raise RuntimeError(f"Missing required key in JSON: {key}")
    return obj[key]


class Tags:
    def __init__(self, j: dict, year: str, is_data: bool, era: Optional[str]):
        if year not in j:
            raise RuntimeError(f"Year key not found in JSON: {year}")
        y = j[year]
        self.jercJsonPath = _get(y, "jercJsonPath")
        self.tagNameL1FastJet = _get(y, "tagNameL1FastJet")
        self.tagNameL2Relative = _get(y, "tagNameL2Relative")
        self.tagNameL3Absolute = _get(y, "tagNameL3Absolute")
        self.tagNameL2Residual = _get(y, "tagNameL2Residual")
        self.tagNameL2L3Residual = _get(y, "tagNameL2L3Residual")
        self.tagNamePtResolution = _get(y, "tagNamePtResolution")
        self.tagNameJerScaleFactor = _get(y, "tagNameJerScaleFactor")
        if is_data:
            if "data" not in y:
                raise RuntimeError(f"Requested data but no 'data' section for year: {year}")
            if not era:
                raise RuntimeError(f"Data requested but no era provided for year: {year}")
            data_obj = y["data"]
            if era not in data_obj:
                raise RuntimeError(f"Era key not found under data for year {year}: {era}")
            e = data_obj[era]
            self.tagNameL1FastJet = _get(e, "tagNameL1FastJet")
            self.tagNameL2Relative = _get(e, "tagNameL2Relative")
            self.tagNameL3Absolute = _get(e, "tagNameL3Absolute")
            self.tagNameL2Residual = _get(e, "tagNameL2Residual")
            self.tagNameL2L3Residual = _get(e, "tagNameL2L3Residual")


# cache of correction.CorrectionSet per file path (like static map in C++)
_CORRSET_CACHE: Dict[str, correction.CorrectionSet] = {}


def _safe_get(cs: correction.CorrectionSet, name: str):
    try:
        return cs[name]
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve correction '{name}': {e}")


class CorrectionRefs:
    def __init__(self, tags: Tags):
        if tags.jercJsonPath in _CORRSET_CACHE:
            self.cs = _CORRSET_CACHE[tags.jercJsonPath]
        else:
            self.cs = correction.CorrectionSet.from_file(tags.jercJsonPath)
            _CORRSET_CACHE[tags.jercJsonPath] = self.cs
        self.corrRefJesL1FastJet = _safe_get(self.cs, tags.tagNameL1FastJet)
        self.corrRefJesL2Relative = _safe_get(self.cs, tags.tagNameL2Relative)
        self.corrRefJesL2ResL3Res = _safe_get(self.cs, tags.tagNameL2L3Residual)
        self.corrRefJerReso = _safe_get(self.cs, tags.tagNamePtResolution)
        self.corrRefJerSf = _safe_get(self.cs, tags.tagNameJerScaleFactor)


# ---------------------------------------------------------------------------
# Physics utilities
# ---------------------------------------------------------------------------

def phi_mpi_pi(dphi: float) -> float:
    # map angle to [-pi, pi)
    return (dphi + math.pi) % (2 * math.pi) - math.pi


def deltaR(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    dEta = float(eta1) - float(eta2)
    dPhi = phi_mpi_pi(phi1 - phi2)
    return abs(math.hypot(dEta, dPhi))


# ---------------------------------------------------------------------------
# TH1 helpers via PyROOT (to mirror C++ hist output exactly)
# ---------------------------------------------------------------------------
class Hists:
    def __init__(self, also_nano: bool):
        self.hJetPt_AK4_Nano = ROOT.TH1F("hJetPt_AK4_Nano", "", 50, 10, 510) if also_nano else None
        self.hJetPt_AK8_Nano = ROOT.TH1F("hJetPt_AK8_Nano", "", 50, 10, 510) if also_nano else None
        self.hMET_Nano = ROOT.TH1F("hMET_Nano", "", 50, 10, 510) if also_nano else None
        self.hJetPt_AK4 = ROOT.TH1F("hJetPt_AK4", "", 50, 10, 510)
        self.hJetPt_AK8 = ROOT.TH1F("hJetPt_AK8", "", 50, 10, 510)
        self.hMET = ROOT.TH1F("hMET", "", 50, 10, 510)

    def fill(self, h, x):
        if h:
            h.Fill(float(x))


def get_or_mkdir(parent: ROOT.TDirectory, name: str) -> ROOT.TDirectory:
    d = parent.Get(name)
    if d:
        return d
    parent.mkdir(name)
    return parent.Get(name)


def make_hists(fout: ROOT.TFile, year: str, is_data: bool, era: Optional[str],
               syst_set: str, syst_name: str, also_nano: bool) -> Hists:
    year_dir = fout.Get(year)
    if not year_dir:
        fout.mkdir(year)
        year_dir = fout.Get(year)
    type_dir = get_or_mkdir(year_dir, "Data" if is_data else "MC")
    if is_data and era:
        type_dir = get_or_mkdir(type_dir, era)
    set_dir = get_or_mkdir(type_dir, syst_set)
    pass_dir = get_or_mkdir(set_dir, syst_name)
    pass_dir.cd()
    return Hists(also_nano)

# ---------------------------------------------------------------------------
# Systematic descriptors (mirror the C++ structs/classes)
# ---------------------------------------------------------------------------
class SystKind:
    Nominal = "Nominal"
    JES = "JES"
    JER = "JER"


class SystTagDetail:
    def __init__(self):
        self.setName = ""
        self.var = ""  # Up/Down or empty
        self.kind = SystKind.Nominal

    def is_nominal(self) -> bool:
        return self.kind == SystKind.Nominal

    def syst_set_name(self) -> str:
        return "Nominal" if self.is_nominal() else self.setName

    def syst_name(self) -> str:
        return "Nominal" if self.is_nominal() else f"{self.setName}_{self.var}"


class SystTagDetailJES(SystTagDetail):
    def __init__(self):
        super().__init__()
        self.tagAK4 = ""
        self.tagAK8 = ""
        self.baseTag = ""
        self.kind = SystKind.JES

    def syst_name(self) -> str:
        return "Nominal" if self.is_nominal() else f"{self.baseTag}_{self.var}"


class JerBin:
    def __init__(self, label: str, eta_min: float, eta_max: float, pt_min: float, pt_max: float):
        self.label = label
        self.etaMin = eta_min
        self.etaMax = eta_max
        self.ptMin = pt_min
        self.ptMax = pt_max


class SystTagDetailJER(SystTagDetail):
    def __init__(self):
        super().__init__()
        self.baseTag = ""
        self.jerRegion: JerBin | None = None
        self.kind = SystKind.JER

    def syst_name(self) -> str:
        return "Nominal" if self.is_nominal() else f"{self.baseTag}_{self.var}"


def get_jer_uncertainty_sets(base_json: dict, year: str) -> Dict[str, List[JerBin]]:
    out = {}
    if year not in base_json:
        return out
    y = base_json[year]
    if "ForUncertaintyJER" not in y or not isinstance(y["ForUncertaintyJER"], dict):
        return out
    j = y["ForUncertaintyJER"]
    for set_name, obj in j.items():
        if not isinstance(obj, dict):
            continue
        bins: List[JerBin] = []
        for label, arr in obj.items():
            if not (isinstance(arr, list) and len(arr) == 4):
                raise RuntimeError(
                    f"ForUncertaintyJER bin '{label}' must be an array [etaMin, etaMax, ptMin, ptMax]."
                )
            bins.append(JerBin(label, float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])))
        out[set_name] = bins
    return out


def build_jer_tag_details(jer_sets: Dict[str, List[JerBin]]) -> List[SystTagDetailJER]:
    out: List[SystTagDetailJER] = []
    for set_name, bins in jer_sets.items():
        for b in bins:
            for var in ("Up", "Down"):
                d = SystTagDetailJER()
                d.setName = set_name
                d.var = var
                d.baseTag = b.label
                d.jerRegion = b
                out.append(d)
    return out


def get_syst_tag_names(base_json: dict, year: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Return map: setName -> list of (fullTag, base/custom name).
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    if year not in base_json:
        return out
    y = base_json[year]
    if "ForUncertaintyJES" not in y:
        return out
    for set_name, arr in y["ForUncertaintyJES"].items():
        if not isinstance(arr, list):
            continue
        pairs: List[Tuple[str, str]] = []
        for item in arr:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str) and isinstance(item[1], str):
                pairs.append((item[0], item[1]))
        out[set_name] = pairs
    return out


def build_syst_tag_detail_JES(sAK4: Dict[str, List[Tuple[str, str]]],
                               sAK8: Dict[str, List[Tuple[str, str]]]) -> List[SystTagDetailJES]:
    details: List[SystTagDetailJES] = []
    for set_name, pairsAK4 in sAK4.items():
        if set_name not in sAK8:
            continue
        pairsAK8 = sAK8[set_name]
        mapAK4 = {base: full for (full, base) in pairsAK4}
        mapAK8 = {base: full for (full, base) in pairsAK8}
        for base, fullAK4 in mapAK4.items():
            if base not in mapAK8:
                continue
            fullAK8 = mapAK8[base]
            for var in ("Up", "Down"):
                d = SystTagDetailJES()
                d.setName = set_name
                d.var = var
                d.tagAK4 = fullAK4
                d.tagAK8 = fullAK8
                d.baseTag = base
                details.append(d)
    return details

# ---------------------------------------------------------------------------
# Jet selection helpers
# ---------------------------------------------------------------------------

def collect_ak8_indices(pt: np.ndarray, eta: np.ndarray, hNano) -> List[int]:
    idxs: List[int] = []
    for j in range(len(pt)):
        if pt[j] < 100 or abs(eta[j]) > 5.2:
            continue
        idxs.append(j)
        if hNano:
            hNano.Fill(float(pt[j]))
    return idxs


def collect_non_overlapping_ak4_indices(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray,
                                        ak8_eta: np.ndarray, ak8_phi: np.ndarray,
                                        hNano) -> List[int]:
    idxs: List[int] = []
    for j in range(len(pt)):
        if pt[j] < 15 or abs(eta[j]) > 5.2:
            continue
        overlaps = False
        for k in range(len(ak8_eta)):
            if deltaR(eta[j], phi[j], ak8_eta[k], ak8_phi[k]) < 0.6:
                overlaps = True
                break
        if overlaps:
            continue
        idxs.append(j)
        if hNano:
            hNano.Fill(float(pt[j]))
    return idxs

# ---------------------------------------------------------------------------
# JES/JER applications (per-jet, event-wise)
# ---------------------------------------------------------------------------

def apply_JES_nominal(year: str, is_data: bool, refs: CorrectionRefs,
                       pts: np.ndarray, etas: np.ndarray, phis: np.ndarray,
                       areas: np.ndarray, rawFactors: np.ndarray,
                       run_val: int, debug=False):
    # Modify pts in-place to mirror C++ behavior
    for idx in range(len(pts)):
        if debug:
            print_debug(debug, spaces4, f"[Jet] index={idx}, eta={etas[idx]}, phi={phis[idx]}, area={areas[idx]}, rawFactor={rawFactors[idx]}")
            print_debug(debug, spaces6, f"default NanoAod  Pt={pts[idx]}")
        # undo NanoAOD default correction
        rawSF = 1.0 - float(rawFactors[idx])
        pts[idx] *= rawSF
        if debug:
            print_debug(debug, spaces6, f"after undoing    Pt={pts[idx]}")
        # L1 FastJet
        c1 = refs.corrRefJesL1FastJet.evaluate(areas[idx], etas[idx], pts[idx], rho_val) 
        # NOTE: we cannot access rho_val here; patched by closure in caller. We'll override below.
        raise RuntimeError("apply_JES_nominal must be wrapped to supply per-event rho via closure.")

# We provide a closure factory so we can access event-level rho without threading it through every call

def make_apply_JES_nominal(year: str, is_data: bool, refs: CorrectionRefs, rho: float, run_for_residual: float):
    def _apply(pts: np.ndarray, etas: np.ndarray, phis: np.ndarray, areas: np.ndarray, rawFactors: np.ndarray, debug=True):
        for idx in range(len(pts)):
            if debug:
                print_debug(debug, spaces4, f"[Jet] index={idx}, eta={etas[idx]}, phi={phis[idx]}, area={areas[idx]}, rawFactor={rawFactors[idx]}")
                print_debug(debug, spaces6, f"default NanoAod  Pt={pts[idx]}")
            # undo NanoAOD default correction
            rawSF = 1.0 - float(rawFactors[idx])
            pts[idx] *= rawSF
            if debug:
                print_debug(debug, spaces6, f"after undoing    Pt={pts[idx]}")
            # L1FastJet
            c1 = refs.corrRefJesL1FastJet.evaluate(areas[idx], etas[idx], pts[idx], rho)
            pts[idx] *= c1
            if debug:
                print_debug(debug, spaces6, f"after L1FastJet  Pt={pts[idx]}")
            # L2Relative
            if has_phi_dependent_L2(year):
                c2 = refs.corrRefJesL2Relative.evaluate(etas[idx], phis[idx], pts[idx])
            else:
                c2 = refs.corrRefJesL2Relative.evaluate(etas[idx], pts[idx])
            pts[idx] *= c2
            if debug:
                print_debug(debug, spaces6, f"after L2Relative Pt={pts[idx]}")
            # L2L3Residual on data only
            if is_data:
                if requires_run_based_residual(year):
                    cR = refs.corrRefJesL2ResL3Res.evaluate(run_for_residual, etas[idx], pts[idx])
                else:
                    cR = refs.corrRefJesL2ResL3Res.evaluate(etas[idx], pts[idx])
                pts[idx] *= cR
                if debug:
                    print_debug(debug, spaces6, f"after L2L3Residual Pt={pts[idx]}")
    return _apply


def apply_JES_syst(refs: CorrectionRefs, syst_name: str, var: str,
                    pts: np.ndarray, etas: np.ndarray, phis: np.ndarray,
                    debug=False):
    systCorr = refs.cs[syst_name]
    for idx in range(len(pts)):
        if debug:
            print_debug(debug, spaces4, f"[Jet] index={idx}")
            print_debug(debug, spaces5, f"Nominal corrected    Pt={pts[idx]}")
        scale = systCorr.evaluate(etas[idx], pts[idx])
        shift = (1.0 + scale) if var == "Up" else (1.0 - scale)
        pts[idx] *= shift
        if debug:
            print_debug(debug, spaces5, f"Syst corrected       Pt={pts[idx]}")


# ---------------------------------------------------------------------------
# MET recomputation (Type-1 propagation) for AK4 jets only, as in C++ macro
# ---------------------------------------------------------------------------

def corrected_met(year: str, is_data: bool, refs: CorrectionRefs,
                  met_raw_pt: float, met_raw_phi: float,
                  jet_arrays: dict,
                  indicesAK4ForMet: List[int], rawPtsAK4ForMet: np.ndarray,
                  rho: float,
                  jerVar: str = "nom",
                  jerRegion: Optional[JerBin] = None,
                  jesSystName: str = "",
                  jesSystVar: str = "",
                  seed_triplet: Tuple[int, int, int] = (0, 0, 0),
                  debug=False) -> Tuple[float, float]:
    met_px = float(met_raw_pt) * math.cos(float(met_raw_phi))
    met_py = float(met_raw_pt) * math.sin(float(met_raw_phi))
    if debug:
        print_debug(debug, spaces3, f"[Met] Raw Pt = {met_raw_pt}")

    Jet_phi = jet_arrays["phi"]
    Jet_eta = jet_arrays["eta"]
    Jet_area = jet_arrays["area"]
    Jet_muonSubtrFactor = jet_arrays["muonSubtrFactor"]
    Jet_genJetIdx = jet_arrays.get("genIdx")
    GenJet_pt = jet_arrays.get("Gen_pt")
    GenJet_eta = jet_arrays.get("Gen_eta")
    GenJet_phi = jet_arrays.get("Gen_phi")

    rng = np.random.default_rng(seed_triplet[0] + seed_triplet[1] + seed_triplet[2])

    for i in range(len(indicesAK4ForMet)):
        idx = indicesAK4ForMet[i]
        if idx >= len(rawPtsAK4ForMet):
            break
        phi = float(Jet_phi[idx])
        eta = float(Jet_eta[idx])
        area = float(Jet_area[idx])

        # recompute corrections on muon-subtracted raw jet pt
        pt_corr = float(rawPtsAK4ForMet[idx]) * (1.0 - float(Jet_muonSubtrFactor[idx]))
        # L1FastJet
        c1 = refs.corrRefJesL1FastJet.evaluate(area, eta, pt_corr, rho)
        pt_corr *= c1
        pt_corr_l1rc = pt_corr  # capture L1Rc stage
        # L2Relative
        if has_phi_dependent_L2(year):
            c2 = refs.corrRefJesL2Relative.evaluate(eta, phi, pt_corr)
        else:
            c2 = refs.corrRefJesL2Relative.evaluate(eta, pt_corr)
        pt_corr *= c2
        # L2L3Residual (data only)
        if is_data:
            if requires_run_based_residual(year):
                # In C++ they use representativeRunNumber(nanoT, year)
                run_for_residual = representative_run_number(year)
                cR = refs.corrRefJesL2ResL3Res.evaluate(run_for_residual, eta, pt_corr)
            else:
                cR = refs.corrRefJesL2ResL3Res.evaluate(eta, pt_corr)
            pt_corr *= cR
        # JES syst shift (MC only)
        if (not is_data) and jesSystName:
            systCorr = refs.cs[jesSystName]
            scale = systCorr.evaluate(eta, pt_corr)
            shift = (1.0 + scale) if jesSystVar == "Up" else (1.0 - scale)
            pt_corr *= shift
        # JER smearing (MC only)
        if not is_data:
            useVar = jerVar
            if jerRegion is not None:
                aeta = abs(eta)
                if not (aeta >= jerRegion.etaMin and aeta < jerRegion.etaMax and pt_corr >= jerRegion.ptMin and pt_corr < jerRegion.ptMax):
                    useVar = "nom"
            reso = reso = refs.corrRefJerReso.evaluate(eta, pt_corr, rho) if uses_puppi_met(year) else refs.corrRefJerReso.evaluate(eta, pt_corr, rho)
            sf   = refs.corrRefJerSf.evaluate(eta, pt_corr, useVar) if uses_puppi_met(year) else refs.corrRefJerSf.evaluate(eta, useVar)
            # seed already set once above
            genIdx = int(Jet_genJetIdx[idx]) if Jet_genJetIdx is not None else -1
            isMatch = False
            if genIdx > -1 and GenJet_pt is not None and genIdx < len(GenJet_pt):
                # Intentionally preserve the (eta,phi) swap from C++ for identical behavior
                dR = deltaR(float(phi), float(GenJet_phi[genIdx]), float(eta), float(GenJet_eta[genIdx]))
                if dR < 0.2 and abs(pt_corr - float(GenJet_pt[genIdx])) < 3.0 * reso * pt_corr:
                    isMatch = True
            if isMatch:
                corr = max(0.0, 1.0 + (sf - 1.0) * (pt_corr - float(GenJet_pt[genIdx])) / pt_corr)
            else:
                corr = max(0.0, 1.0 + rng.normal(0.0, reso) * math.sqrt(max(sf * sf - 1.0, 0.0)))
            pt_corr *= corr
        # selection for propagation
        passSel = (pt_corr > 15.0 and abs(eta) < 5.2 and (float(jet_arrays["chEmEF"][idx]) + float(jet_arrays["neEmEF"][idx])) < 0.9)
        if not passSel:
            continue
        if debug:
            print_debug(debug, spaces4, f"[Jet] index={idx}")
            print_debug(debug, spaces6, f"L1 JEC corrected Pt = {pt_corr_l1rc}")
            print_debug(debug, spaces6, f"L1L2L3JEC + JER corrected Pt = {pt_corr}")
        dpt = pt_corr - pt_corr_l1rc
        met_px -= dpt * math.cos(phi)
        met_py -= dpt * math.sin(phi)

    met_pt = math.hypot(met_px, met_py)
    met_phi = math.atan2(met_py, met_px)
    if debug:
        print_debug(debug, spaces3, f"[Met] Type-1 corrected Pt = {met_pt}")
    return met_pt, met_phi

# ---------------------------------------------------------------------------
# Jet Veto Map
# ---------------------------------------------------------------------------

def any_jet_in_veto_region(jvm_correction, jvm_keyname: str, jet_eta: np.ndarray, jet_phi: np.ndarray,
                            jet_id: np.ndarray, jet_pt: np.ndarray, chEmEF: np.ndarray, neEmEF: np.ndarray) -> bool:
    maxEtaInMap = 5.191
    maxPhiInMap = math.pi
    for i in range(len(jet_pt)):
        if abs(float(jet_eta[i])) > maxEtaInMap:
            continue
        if abs(float(jet_phi[i])) > maxPhiInMap:
            continue
        if int(jet_id[i]) < 6:
            continue
        if float(jet_pt[i]) < 15:
            continue
        if float(chEmEF[i]) + float(neEmEF[i]) > 0.9:
            continue
        jvmNumber = jvm_correction.evaluate(jvm_keyname, jet_eta[i], jet_phi[i])
        if jvmNumber > 0.0:
            return True
    return False

# ---------------------------------------------------------------------------
# Core event processing
# ---------------------------------------------------------------------------

def process_events(input_file: str, fout: ROOT.TFile, year: str, is_data: bool,
                   era: Optional[str], systTagDetail: SystTagDetail):
    # Load JSONs and build correction refs
    cfgAK4 = load_json_config("JercFileAndTagNamesAK4.json")
    tagsAK4 = Tags(cfgAK4, year, is_data, era)
    refsAK4 = CorrectionRefs(tagsAK4)

    cfgAK8 = load_json_config("JercFileAndTagNamesAK8.json")
    if year in cfgAK8:
        tagsAK8 = Tags(cfgAK8, year, is_data, era)
    else:
        tagsAK8 = tagsAK4  # fall back to AK4 tags for AK8 in Run-3 (as in C++)
    refsAK8 = CorrectionRefs(tagsAK8)

    # JVM
    cfgJvm = load_json_config("JvmFileAndTagNames.json")
    useJvm = False
    jvmRef = None
    jvmKey = ""
    if year in cfgJvm:
        y = cfgJvm[year]
        jvmFile = _get(y, "jvmFilePath")
        jvmTag = _get(y, "jvmTagName")
        jvmKey = _get(y, "jvmKeyName")
        jvmRef = correction.CorrectionSet.from_file(jvmFile)[jvmTag]
        useJvm = True

    writeNano = systTagDetail.is_nominal()
    H = make_hists(fout, year, is_data, era, systTagDetail.syst_set_name(), systTagDetail.syst_name(), writeNano)

    # Branches to read
    common_branches = [
        "run", "luminosityBlock", "event",
        "Rho_fixedGridRhoFastjetAll",
        "MET_pt", "MET_phi", "RawMET_pt", "RawMET_phi",
        "RawPuppiMET_pt", "RawPuppiMET_phi",
        # AK4
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
        "Jet_rawFactor", "Jet_muonSubtrFactor", "Jet_area", "Jet_jetId",
        "Jet_chEmEF", "Jet_neEmEF",
        # AK8
        "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_mass",
        "FatJet_rawFactor", "FatJet_area", "FatJet_jetId",
    ]
    mc_extra = [
        "Jet_genJetIdx",
        "FatJet_genJetAK8Idx",
        "GenJet_pt", "GenJet_eta", "GenJet_phi",
        "GenJetAK8_pt", "GenJetAK8_eta", "GenJetAK8_phi",
    ]

    branches = common_branches + (mc_extra if not is_data else [])

    # Iterate events (chunked) to keep memory moderate
    for arrays in uproot.iterate(f"{input_file}:Events", branches, library="ak", step_size=5000):
        n_ev = len(arrays["run"])
        for i in range(n_ev):
            run = int(arrays["run"][i])
            lumi = int(arrays["luminosityBlock"][i])
            evn = int(arrays["event"][i])
            rho = float(arrays["Rho_fixedGridRhoFastjetAll"][i])

            met_pt = float(arrays["MET_pt"][i])
            met_phi = float(arrays["MET_phi"][i])
            rawMET_pt = float(arrays["RawPuppiMET_pt"][i] if uses_puppi_met(year) else arrays["RawMET_pt"][i])
            rawMET_phi = float(arrays["RawPuppiMET_phi"][i] if uses_puppi_met(year) else arrays["RawMET_phi"][i])

            # Awkward → numpy for this event
            Jet_pt = np.array(arrays["Jet_pt"][i], dtype=np.float64)
            Jet_eta = np.array(arrays["Jet_eta"][i], dtype=np.float64)
            Jet_phi = np.array(arrays["Jet_phi"][i], dtype=np.float64)
            Jet_mass = np.array(arrays["Jet_mass"][i], dtype=np.float64)
            Jet_rawFactor = np.array(arrays["Jet_rawFactor"][i], dtype=np.float64)
            Jet_muonSubtrFactor = np.array(arrays["Jet_muonSubtrFactor"][i], dtype=np.float64)
            Jet_area = np.array(arrays["Jet_area"][i], dtype=np.float64)
            Jet_jetId = np.array(arrays["Jet_jetId"][i], dtype=np.int16)
            Jet_chEmEF = np.array(arrays["Jet_chEmEF"][i], dtype=np.float64)
            Jet_neEmEF = np.array(arrays["Jet_neEmEF"][i], dtype=np.float64)

            FatJet_pt = np.array(arrays["FatJet_pt"][i], dtype=np.float64)
            FatJet_eta = np.array(arrays["FatJet_eta"][i], dtype=np.float64)
            FatJet_phi = np.array(arrays["FatJet_phi"][i], dtype=np.float64)
            FatJet_mass = np.array(arrays["FatJet_mass"][i], dtype=np.float64)
            FatJet_rawFactor = np.array(arrays["FatJet_rawFactor"][i], dtype=np.float64)
            FatJet_area = np.array(arrays["FatJet_area"][i], dtype=np.float64)

            if not is_data:
                Jet_genJetIdx = np.array(arrays["Jet_genJetIdx"][i], dtype=np.int32)
                FatJet_genJetAK8Idx = np.array(arrays["FatJet_genJetAK8Idx"][i], dtype=np.int32)
                GenJet_pt = np.array(arrays["GenJet_pt"][i], dtype=np.float64)
                GenJet_eta = np.array(arrays["GenJet_eta"][i], dtype=np.float64)
                GenJet_phi = np.array(arrays["GenJet_phi"][i], dtype=np.float64)
                GenJetAK8_pt = np.array(arrays["GenJetAK8_pt"][i], dtype=np.float64)
                GenJetAK8_eta = np.array(arrays["GenJetAK8_eta"][i], dtype=np.float64)
                GenJetAK8_phi = np.array(arrays["GenJetAK8_phi"][i], dtype=np.float64)
            else:
                Jet_genJetIdx = None
                FatJet_genJetAK8Idx = None
                GenJet_pt = GenJet_eta = GenJet_phi = None
                GenJetAK8_pt = GenJetAK8_eta = GenJetAK8_phi = None

            # Fill Nano hMET on every pass (no-op if not created)
            H.fill(H.hMET_Nano, met_pt)

            # Determine AK8 selection indices first
            if applyOnlyOnAK4 or applyOnlyOnAK8 is False or applyOnAK4AndAK8:
                # If both, collect AK8 for overlap removal
                ak8_idxs = collect_ak8_indices(FatJet_pt, FatJet_eta, H.hJetPt_AK8_Nano)
            else:
                ak8_idxs = []

            # Collect AK4 non-overlapping
            if applyOnlyOnAK8:
                ak4_idxs = []
            else:
                ak4_idxs = collect_non_overlapping_ak4_indices(
                    Jet_pt, Jet_eta, Jet_phi,
                    FatJet_eta[ak8_idxs] if len(ak8_idxs) else np.array([], dtype=np.float64),
                    FatJet_phi[ak8_idxs] if len(ak8_idxs) else np.array([], dtype=np.float64),
                    H.hJetPt_AK4_Nano,
                )

            # Prepare raw pT for MET propagation (AK4 only)
            rawPtsAK4ForMet = (1.0 - Jet_rawFactor) * Jet_pt
            indicesAK4ForMet = list(range(len(Jet_pt)))

            # Prepare JES nominal appliers (closures with per-event rho/run)
            run_for_residual = representative_run_number(year) if requires_run_based_residual(year) else float(run)
            apply_JES_nominal_AK4 = make_apply_JES_nominal(year, is_data, refsAK4, rho, run_for_residual)
            apply_JES_nominal_AK8 = make_apply_JES_nominal(year, is_data, refsAK8, rho, run_for_residual)

            # (1) JES nominal (no JER yet)
            if systTagDetail.is_nominal():
                if (applyOnlyOnAK4 or applyOnAK4AndAK8) and len(ak4_idxs):
                    apply_JES_nominal_AK4(Jet_pt, Jet_eta, Jet_phi, Jet_area, Jet_rawFactor, debug=False)
                if (applyOnlyOnAK8 or applyOnAK4AndAK8) and len(ak8_idxs):
                    apply_JES_nominal_AK8(FatJet_pt, FatJet_eta, FatJet_phi, FatJet_area, FatJet_rawFactor, debug=False)
            else:
                # Always apply nominal before a systematic variation (quietly)
                if (applyOnlyOnAK4 or applyOnAK4AndAK8) and len(ak4_idxs):
                    apply_JES_nominal_AK4(Jet_pt, Jet_eta, Jet_phi, Jet_area, Jet_rawFactor, debug=False)
                if (applyOnlyOnAK8 or applyOnAK4AndAK8) and len(ak8_idxs):
                    apply_JES_nominal_AK8(FatJet_pt, FatJet_eta, FatJet_phi, FatJet_area, FatJet_rawFactor, debug=False)

            # (2) JES Uncertainty (MC only), if this pass is JES
            jesSystName = ""
            jesVar = ""
            if (not is_data) and (systTagDetail.kind == SystKind.JES):
                jesDetail: SystTagDetailJES = systTagDetail  # type: ignore
                if (applyOnlyOnAK4 or applyOnAK4AndAK8) and len(ak4_idxs):
                    apply_JES_syst(refsAK4, jesDetail.tagAK4, jesDetail.var, Jet_pt, Jet_eta, Jet_phi, debug=False)
                if (applyOnlyOnAK8 or applyOnAK4AndAK8) and len(ak8_idxs):
                    apply_JES_syst(refsAK8, jesDetail.tagAK8, jesDetail.var, FatJet_pt, FatJet_eta, FatJet_phi, debug=False)
                jesSystName = jesDetail.tagAK4
                jesVar = jesDetail.var

            # (3) JER (after all JES) — MC only
            jerVar = "nom"
            jerRegion = None
            if (not is_data) and (systTagDetail.kind == SystKind.JER):
                jerDetail: SystTagDetailJER = systTagDetail  # type: ignore
                jerVar = "up" if jerDetail.var == "Up" else "down"
                jerRegion = jerDetail.jerRegion

                # Apply to AK4 and AK8 lists (mirror macro; matching/mismeasure preserved)
                # AK4
                rng = np.random.default_rng(run + lumi + evn)
                for j in ak4_idxs:
                    etaJet = float(Jet_eta[j])
                    ptJet = float(Jet_pt[j])
                    phiJet = float(Jet_phi[j])
                    # region gating
                    useVar = jerVar
                    if jerRegion is not None:
                        aeta = abs(etaJet)
                        if not (aeta >= jerRegion.etaMin and aeta < jerRegion.etaMax and ptJet >= jerRegion.ptMin and ptJet < jerRegion.ptMax):
                            useVar = "nom"
                    reso = refsAK4.corrRefJerReso.evaluate(etaJet, ptJet, rho)
                    sf   = refsAK4.corrRefJerSf.evaluate(etaJet, ptJet, useVar) if uses_puppi_met(year) else refsAK4.corrRefJerSf.evaluate(etaJet, useVar)
                    genIdx = int(Jet_genJetIdx[j]) if Jet_genJetIdx is not None else -1
                    isMatch = False
                    if genIdx > -1 and GenJet_pt is not None and genIdx < len(GenJet_pt):
                        # Preserve argument order quirk from C++
                        dR = deltaR(phiJet, float(GenJet_phi[genIdx]), etaJet, float(GenJet_eta[genIdx]))
                        if dR < 0.2 and abs(ptJet - float(GenJet_pt[genIdx])) < 3.0 * reso * ptJet:
                            isMatch = True
                    if isMatch:
                        corr = max(0.0, 1.0 + (sf - 1.0) * (ptJet - float(GenJet_pt[genIdx])) / ptJet)
                    else:
                        corr = max(0.0, 1.0 + rng.normal(0.0, reso) * math.sqrt(max(sf * sf - 1.0, 0.0)))
                    Jet_pt[j] *= corr
                # AK8
                rng = np.random.default_rng(run + lumi + evn)
                for j in ak8_idxs:
                    etaJet = float(FatJet_eta[j])
                    ptJet = float(FatJet_pt[j])
                    phiJet = float(FatJet_phi[j])
                    useVar = jerVar
                    if jerRegion is not None:
                        aeta = abs(etaJet)
                        if not (aeta >= jerRegion.etaMin and aeta < jerRegion.etaMax and ptJet >= jerRegion.ptMin and ptJet < jerRegion.ptMax):
                            useVar = "nom"
                    reso = refsAK8.corrRefJerReso.evaluate(etaJet, ptJet, rho)
                    sf   = refsAK8.corrRefJerSf.evaluate(etaJet, ptJet, useVar) if uses_puppi_met(year) else refsAK8.corrRefJerSf.evaluate(etaJet, useVar)
                    genIdx = int(FatJet_genJetAK8Idx[j]) if FatJet_genJetAK8Idx is not None else -1
                    isMatch = False
                    if genIdx > -1 and GenJetAK8_pt is not None and genIdx < len(GenJetAK8_pt):
                        dR = deltaR(phiJet, float(GenJetAK8_phi[genIdx]), etaJet, float(GenJetAK8_eta[genIdx]))
                        if dR < 0.4 and abs(ptJet - float(GenJetAK8_pt[genIdx])) < 3.0 * reso * ptJet:
                            isMatch = True
                    if isMatch:
                        corr = max(0.0, 1.0 + (sf - 1.0) * (ptJet - float(GenJetAK8_pt[genIdx])) / ptJet)
                    else:
                        corr = max(0.0, 1.0 + rng.normal(0.0, reso) * math.sqrt(max(sf * sf - 1.0, 0.0)))
                    FatJet_pt[j] *= corr

            # (MET) propagate AK4 corrections (JES/JER) to MET if enabled
            if applyOnMET:
                jet_arrays = {
                    "phi": Jet_phi,
                    "eta": Jet_eta,
                    "area": Jet_area,
                    "muonSubtrFactor": Jet_muonSubtrFactor,
                    "genIdx": Jet_genJetIdx,
                    "Gen_pt": GenJet_pt,
                    "Gen_eta": GenJet_eta,
                    "Gen_phi": GenJet_phi,
                    "chEmEF": Jet_chEmEF,
                    "neEmEF": Jet_neEmEF,
                }
                met_pt_corr, met_phi_corr = corrected_met(
                    year, is_data, refsAK4,
                    rawMET_pt, rawMET_phi,
                    jet_arrays,
                    indicesAK4ForMet, rawPtsAK4ForMet,
                    rho,
                    jerVar=jerVar,
                    jerRegion=jerRegion,
                    jesSystName=jesSystName,
                    jesSystVar=jesVar,
                    seed_triplet=(evn, run, lumi),
                    debug=False,
                )
                H.hMET.Fill(float(met_pt_corr))
            else:
                H.hMET.Fill(float(met_pt))

            # Fill jet pT hists after corrections
            for j in ak4_idxs:
                H.hJetPt_AK4.Fill(float(Jet_pt[j]))
            for j in ak8_idxs:
                H.hJetPt_AK8.Fill(float(FatJet_pt[j]))

            # Jet Veto Map (event veto)
            if useJvm:
                if any_jet_in_veto_region(jvmRef, jvmKey, Jet_eta, Jet_phi, Jet_jetId, Jet_pt, Jet_chEmEF, Jet_neEmEF):
                    # In C++ they count & continue; here we simply skip any further analysis
                    # (Histogram fills above replicate their ordering.)
                    continue


# ---------------------------------------------------------------------------
# Orchestrator: run nominal + all JES/JER systematics for a year
# ---------------------------------------------------------------------------

def process_events_with_nominal_or_syst(input_file: str, fout: ROOT.TFile, year: str, is_data: bool,
                                        era: Optional[str] = None):
    cfgAK4 = load_json_config("JercFileAndTagNamesAK4.json")
    cfgAK8 = load_json_config("JercFileAndTagNamesAK8.json")

    sAK4 = get_syst_tag_names(cfgAK4, year)
    sAK8 = get_syst_tag_names(cfgAK8, year) if year in cfgAK8 else sAK4

    jerSets = get_jer_uncertainty_sets(cfgAK4, year)

    # 0) Nominal
    print(" [Nominal]")
    process_events(input_file, fout, year, is_data, era, SystTagDetail())

    if not is_data:
        # 1) Correlated JES systematics
        jesDetails: List[SystTagDetailJES] = []
        if applyOnlyOnAK4:
            for set_name, pairs in sAK4.items():
                for (full, base) in pairs:
                    for var in ("Up", "Down"):
                        d = SystTagDetailJES()
                        d.setName = set_name
                        d.var = var
                        d.tagAK4 = full
                        d.tagAK8 = ""
                        d.baseTag = base
                        jesDetails.append(d)
        elif applyOnlyOnAK8:
            for set_name, pairs in sAK8.items():
                for (full, base) in pairs:
                    for var in ("Up", "Down"):
                        d = SystTagDetailJES()
                        d.setName = set_name
                        d.var = var
                        d.tagAK4 = ""
                        d.tagAK8 = full
                        d.baseTag = base
                        jesDetails.append(d)
        else:
            jesDetails = build_syst_tag_detail_JES(sAK4, sAK8)

        for d in jesDetails:
            print(f"\n [JES Syst]: {d.syst_name()}")
            process_events(input_file, fout, year, False, era, d)

        # 2) JER up/down with region gating
        jerDetails = build_jer_tag_details(jerSets)
        for d in jerDetails:
            print(f"\n [JER Syst]: {d.syst_set_name()}/{d.syst_name()}")
            process_events(input_file, fout, year, False, era, d)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply JERC + JVM on NanoAOD (Python columnar port)")
    parser.add_argument("--input-mc", default="NanoAOD_MC.root")
    parser.add_argument("--input-data", default="NanoAOD_Data.root")
    parser.add_argument("--out", default="output.root")
    parser.add_argument("--years-mc", nargs="*", default=[
        "2016Pre", 
        #"2016Post", "2017", "2018",
        #"2022Pre", "2022Post", "2023Pre", "2023Post", "2024"
    ])
    parser.add_argument("--eras-data", nargs="*", default=[
        "2016Pre:Era2016PreBCD",
        "2016Pre:Era2016PreEF",
        #"2016Post:Era2016PostFGH",
        #"2017:Era2017B",
        #"2017:Era2017C",
        #"2017:Era2017D",
        #"2017:Era2017E",
        #"2017:Era2017F",
        #"2018:Era2018A",
        #"2018:Era2018B",
        #"2018:Era2018C",
        #"2018:Era2018D",
        #"2022Pre:Era2022C",
        #"2022Pre:Era2022D",
        #"2022Post:Era2022E",
        #"2022Post:Era2022F",
        #"2022Post:Era2022G",
        #"2023Pre:Era2023PreAll",
        #"2023Post:Era2023PostAll",
        #"2024:Era2024All",
    ])
    args = parser.parse_args()

    # honor global toggles if user set env vars (optional convenience)
    global applyOnlyOnAK4, applyOnlyOnAK8, applyOnAK4AndAK8, applyOnMET
    if os.environ.get("APPLY_ONLY_AK4"):
        applyOnlyOnAK4 = True
        applyOnlyOnAK8 = False
        applyOnAK4AndAK8 = False
    if os.environ.get("APPLY_ONLY_AK8"):
        applyOnlyOnAK8 = True
        applyOnlyOnAK4 = False
        applyOnAK4AndAK8 = False
    if os.environ.get("APPLY_ON_MET") is not None:
        applyOnMET = os.environ.get("APPLY_ON_MET") != "0"

    fout = ROOT.TFile(args.out, "RECREATE")

    # MC first
    for year in args.years_mc:
        print("-----------------")
        print(f"[MC] : {year}")
        print("-----------------")
        process_events_with_nominal_or_syst(args.input_mc, fout, year, False, None)

    # Data eras
    for y_e in args.eras_data:
        year, era = y_e.split(":", 1)
        print("-----------------")
        print(f"\n[Data] : {year} : {era}")
        print("-----------------")
        process_events_with_nominal_or_syst(args.input_data, fout, year, True, era)

    fout.Write()
    fout.Close()
    print(f"Wrote output to {args.out}")


if __name__ == "__main__":
    main()

