#!/usr/bin/env python3
# applyJercAndJvm_awkd.py
#
# Columnar port of applyJercAndJvm.C using uproot + awkward + correctionlib.
# PyROOT is used only to write output histograms in the same structure.

import os
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import awkward as ak
import uproot
import vector
import correctionlib._core as correction

# Only for writing histograms/dirs with familiar ROOT layout
import ROOT

try:
    from tqdm.auto import tqdm  # type: ignore
    _TQDM_AVAILABLE = True

    def _progress_print(message: str) -> None:
        """Print messages without breaking tqdm progress bars."""
        tqdm.write(message)

except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None  # type: ignore
    _TQDM_AVAILABLE = False

    def _progress_print(message: str) -> None:
        print(message)

# ---------------------------
# Global toggles (match C++ defaults)
# ---------------------------
applyOnlyOnAK4 = True
applyOnlyOnAK8 = False
applyOnAK4AndAK8 = False
applyOnMET = True
isDebug = True

# ---------------------------
# Helpers / small utilities
# ---------------------------

from functools import lru_cache


def printDebug(*args, indent: int = 0, enable: Optional[bool] = None) -> None:
    """Conditionally print debug information with optional indentation."""
    if enable is None:
        enable = isDebug
    if not enable:
        return

    prefix = " " * max(indent, 0)
    message = " ".join(str(arg) for arg in args)
    if prefix:
        print(prefix + message)
    else:
        print(message)


def _preview_values(arr, limit: int = 5):
    """Return a small list of representative values from an array-like object."""
    if limit <= 0:
        return []

    if isinstance(arr, ak.Array):
        sample = arr[: max(1, limit)]
        try:
            flat = ak.flatten(sample, axis=None)
            np_arr = ak.to_numpy(flat)
        except Exception:
            try:
                as_list = ak.to_list(sample)
            except Exception:
                return []

            collected = []

            def collect(obj):
                if len(collected) >= limit:
                    return
                if isinstance(obj, list):
                    for item in obj:
                        if len(collected) >= limit:
                            break
                        collect(item)
                else:
                    collected.append(obj)

            collect(as_list)
            return collected
    else:
        np_arr = np.asarray(arr)

    if np_arr.size == 0:
        return []

    flat_np = np_arr.reshape(-1)
    if flat_np.size == 0:
        return []
    return [round(v, 2) for v in flat_np[:limit].tolist()]


def _extract_event_slice(arr, event_index: int, mask=None):
    """Return the slice of ``arr`` corresponding to ``event_index`` and optional ``mask``."""

    if event_index is None:
        return arr

    data = arr
    if isinstance(arr, ak.Array):
        if len(arr) <= event_index:
            return arr
        data = arr[event_index]
        if mask is not None:
            mask_arr = mask[event_index]
            try:
                data = data[mask_arr]
            except Exception:
                pass
    else:
        np_arr = np.asarray(arr)
        if np_arr.ndim == 0:
            data = np_arr
        elif event_index < np_arr.shape[0]:
            data = np_arr[event_index]
            if mask is not None:
                mask_arr = np.asarray(mask)
                try:
                    mask_slice = mask_arr[event_index]
                    data = data[mask_slice]
                except Exception:
                    pass
        else:
            data = np_arr
    return data


def debug_values(
    label: str,
    arr,
    indent: int = 0,
    limit: int = 5,
    event_index: Optional[int] = None,
    mask=None,
) -> None:
    """Helper to print a label and a preview of array values when debugging."""
    values = _preview_values(_extract_event_slice(arr, event_index, mask=mask), limit=limit)
    printDebug(f"{label}", values, indent=indent)


def _find_debug_event_index(
    ak4_sel,
    ak8_sel,
    apply_only_ak4: bool,
    apply_only_ak8: bool,
    apply_both: bool,
) -> Optional[int]:
    """Return index of an event suitable for debug printing based on jet selections."""

    if not isDebug:
        return None

    n_events = len(ak4_sel) if ak4_sel is not None else len(ak8_sel)
    if n_events == 0:
        return None

    def to_mask(selection):
        if selection is None:
            return np.zeros(n_events, dtype=bool)
        if len(selection) == 0:
            return np.zeros(n_events, dtype=bool)
        try:
            mask = ak.to_numpy(ak.any(selection, axis=1))
        except Exception:
            mask = np.zeros(n_events, dtype=bool)
        if mask.shape[0] != n_events:
            mask = np.resize(mask, n_events)
        return mask

    ak4_mask = to_mask(ak4_sel)
    ak8_mask = to_mask(ak8_sel)

    if apply_both:
        candidate_mask = ak4_mask & ak8_mask
    elif apply_only_ak4:
        candidate_mask = ak4_mask
    elif apply_only_ak8:
        candidate_mask = ak8_mask
    else:
        candidate_mask = ak4_mask | ak8_mask

    candidate_idx = np.nonzero(candidate_mask)[0]
    if candidate_idx.size == 0:
        return None

    return int(candidate_idx[0])

@lru_cache(maxsize=None)
def load_json_cached(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

class CorrCache:
    """Caches tag JSONs, CorrectionRefs, and JVM per (year,isData,era)."""
    _cfg = {}     # kind -> parsed tag JSON (AK4/AK8)
    _refs = {}    # (kind, year, isData, era_str) -> CorrectionRefs
    _jvm  = {}    # year -> (jvm_ref, jvm_key) or (None, None)

    @classmethod
    def get_cfg(cls, kind: str) -> dict:
        if kind not in cls._cfg:
            path = f"JercFileAndTagNames{kind}.json"
            cls._cfg[kind] = load_json_cached(path)
        return cls._cfg[kind]

    @classmethod
    def get_refs(cls, kind: str, year: str, isData: bool, era: Optional[str]) -> "CorrectionRefs":
        key = (kind, year, bool(isData), era or "")
        if key not in cls._refs:
            cfg = cls.get_cfg(kind)
            # Fallback for AK8 tags (reuse AK4 tags if AK8 not present for this year)
            if kind == "AK8" and year not in cfg:
                cfg = cls.get_cfg("AK4")
            tags = get_tag_names(cfg, year, isData, era)
            cls._refs[key] = CorrectionRefs(tags)
        return cls._refs[key]

    @classmethod
    def get_jvm(cls, year: str):
        if year not in cls._jvm:
            cfgJvm = load_json_cached("JvmFileAndTagNames.json")
            if year in cfgJvm:
                cs = correction.CorrectionSet.from_file(cfgJvm[year]["jvmFilePath"])
                cls._jvm[year] = (cs[cfgJvm[year]["jvmTagName"]], cfgJvm[year]["jvmKeyName"])
            else:
                cls._jvm[year] = (None, None)
        return cls._jvm[year]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def phi_mpi_pi(dphi):
    """Wrap delta-phi into [-pi, pi]."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))

def deltaR(eta1, phi1, eta2, phi2):
    """Vectorized deltaR (supports numpy/awkward)."""
    dEta = eta1 - eta2
    dPhi = phi_mpi_pi(phi1 - phi2)
    return np.sqrt(dEta * dEta + dPhi * dPhi)

# ---------- CORRECTIONLIB SAFE EVAL HELPERS (works even without numpy-vectorized build) ----------

def _broadcast_to_common_shape(*cols):
    """Ensure Awkward, broadcast to common jagged shape, return broadcast cols + counts per event."""
    ak_cols = [c if isinstance(c, ak.Array) else ak.Array(c) for c in cols]
    bcols = ak.broadcast_arrays(*ak_cols)
    counts = ak.num(bcols[0], axis=1)  # per-event object counts (jets etc.)
    return bcols, counts

def _eval_numeric(corr, *cols, blocksize=200_000):
    """
    Numeric-only node. Broadcast with Awkward, flatten to NumPy float64, then:
      - try vectorized evaluate on chunks;
      - if the build doesn't support it, fall back to scalar evaluate in that chunk.
    """
    bcols, counts = _broadcast_to_common_shape(*cols)

    flats = [ak.to_numpy(ak.flatten(x, axis=None)).astype(np.float64, copy=False) for x in bcols]
    n = flats[0].size
    if n == 0:
        return ak.zeros_like(bcols[0], dtype=np.float64)

    out = np.empty(n, dtype=np.float64)

    # process in chunks to keep memory/timing sane
    for start in range(0, n, blocksize):
        end = min(start + blocksize, n)
        sl = slice(start, end)
        chunk_args = [a[sl] for a in flats]
        try:
            # Try vectorized evaluate if available
            out[sl] = np.asarray(corr.evaluate(*chunk_args), dtype=np.float64)
        except Exception:
            # Fallback: scalar loop (pybind always supports scalars)
            for i in range(start, end):
                out[i] = float(corr.evaluate(*(a[i] for a in flats)))

    return ak.unflatten(out, counts)

def _eval_with_string_last(corr, *num_cols, cat, blocksize=200_000):
    """
    String categorical argument comes LAST in the correction signature:
      e.g. (eta, pt, 'up'/'down')
    """
    bcols, counts = _broadcast_to_common_shape(*num_cols)
    flats = [ak.to_numpy(ak.flatten(x, axis=None)).astype(np.float64, copy=False) for x in bcols]
    n = flats[0].size if flats else 0
    if n == 0:
        return ak.zeros_like(bcols[0] if bcols else ak.Array([[]]), dtype=np.float64)

    if isinstance(cat, str):
        cats = np.full(n, cat, dtype=object)
    else:
        bcat, _ = _broadcast_to_common_shape(cat, bcols[0])
        cats = ak.to_numpy(ak.flatten(bcat[0], axis=None)).astype(object)

    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, blocksize):
        end = min(start + blocksize, n)
        for i in range(start, end):
            args = [a[i] for a in flats] + [cats[i]]
            out[i] = float(corr.evaluate(*args))
    return ak.unflatten(out, counts)


def _eval_with_string_first(corr, *num_cols, cat, blocksize=200_000):
    """
    String categorical argument comes FIRST in the correction signature:
      e.g. ('map_key', eta, phi)
    """
    bcols, counts = _broadcast_to_common_shape(*num_cols)
    flats = [ak.to_numpy(ak.flatten(x, axis=None)).astype(np.float64, copy=False) for x in bcols]
    n = flats[0].size if flats else 0
    if n == 0:
        return ak.zeros_like(bcols[0] if bcols else ak.Array([[]]), dtype=np.float64)

    if isinstance(cat, str):
        cats = np.full(n, cat, dtype=object)
    else:
        bcat, _ = _broadcast_to_common_shape(cat, bcols[0])
        cats = ak.to_numpy(ak.flatten(bcat[0], axis=None)).astype(object)

    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, blocksize):
        end = min(start + blocksize, n)
        for i in range(start, end):
            args = [cats[i]] + [a[i] for a in flats]
            out[i] = float(corr.evaluate(*args))
    return ak.unflatten(out, counts)


def take_by_index(arr, idx, fill=0.0):
    valid = (idx >= 0) & (idx < ak.num(arr, axis=1))
    idx_safe = ak.where(valid, idx, 0)
    picked = arr[idx_safe]
    return ak.where(valid, picked, fill), valid

def representative_run_number(year: str, runs):
    """Use a representative run for run-based residuals (as in C++)."""
    if year == "2023Pre":
        return ak.ones_like(runs, dtype="float64") * 367080.0
    if year == "2023Post":
        return ak.ones_like(runs, dtype="float64") * 369803.0
    if year == "2024":
        return ak.ones_like(runs, dtype="float64") * 379412.0
    return ak.values_astype(runs, "float64")

def has_phi_dependent_L2(year: str) -> bool:
    return year in ("2023Post", "2024")

def requires_run_based_residual(year: str) -> bool:
    return year in ("2023Pre", "2023Post", "2024")

def uses_puppi_met(year: str) -> bool:
    return year in {"2022Pre","2022Post","2023Pre","2023Post","2024"}

def is_mc(isData: bool) -> bool:
    return not isData

# Deterministic per-jet standard normal via hashing (Box–Muller)
def normal_from_seeds(seedA, seedB):
    # seeds should be unsigned 64-bit integers (awkward/numpy)
    UINT64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
    a = (seedA ^ np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9) & UINT64_MAX
    b = (seedB ^ np.uint64(0x94D049BB133111EB)) * np.uint64(0xD2B74407B1CE6E93) & UINT64_MAX
    # map to (0,1); avoid 0
    u1 = (a.astype(np.float64) + 1.0) / (UINT64_MAX.astype(np.float64) + 2.0)
    u2 = (b.astype(np.float64) + 1.0) / (UINT64_MAX.astype(np.float64) + 2.0)
    # Box–Muller
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z

# ---------------------------
# Config tag containers
# ---------------------------
@dataclass
class Tags:
    jercJsonPath: str
    tagNameL1FastJet: str
    tagNameL2Relative: str
    tagNameL3Absolute: str
    tagNameL2Residual: str
    tagNameL2L3Residual: str
    tagNamePtResolution: str
    tagNameJerScaleFactor: str

def get_tag_names(base: dict, year: str, isData: bool, era: Optional[str]) -> Tags:
    if year not in base:
        raise RuntimeError(f"Year key not found in JSON: {year}")
    y = base[year].copy()
    # For data, override from era section
    if isData:
        if "data" not in y:
            raise RuntimeError(f"Requested data but no 'data' section for year: {year}")
        if era is None:
            raise RuntimeError(f"Data requested but no era provided for year: {year}")
        dy = y["data"][era]
        y.update(dy)
    return Tags(
        jercJsonPath=y["jercJsonPath"],
        tagNameL1FastJet=y["tagNameL1FastJet"],
        tagNameL2Relative=y["tagNameL2Relative"],
        tagNameL3Absolute=y["tagNameL3Absolute"],
        tagNameL2Residual=y["tagNameL2Residual"],
        tagNameL2L3Residual=y["tagNameL2L3Residual"],
        tagNamePtResolution=y["tagNamePtResolution"],
        tagNameJerScaleFactor=y["tagNameJerScaleFactor"],
    )

class CorrectionRefs:
    def __init__(self, tags: Tags):
        self.cs = correction.CorrectionSet.from_file(tags.jercJsonPath)
        self.cL1 = self.cs[tags.tagNameL1FastJet]
        self.cL2 = self.cs[tags.tagNameL2Relative]
        # Your C++ uses a unified L2L3Residual node (covers residuals)
        self.cL2L3Res = self.cs[tags.tagNameL2L3Residual]
        self.cReso = self.cs[tags.tagNamePtResolution]
        self.cJerSF = self.cs[tags.tagNameJerScaleFactor]

# ---------------------------
# JES nominal (vectorized)
# ---------------------------
def jes_nominal(pt_raw, eta, phi, area, rho, year: str, isData: bool, runs, refs: CorrectionRefs):
    """
    Starting from raw (pT undo), apply L1, L2, (residuals on data).
    Returns corrected pT and pT right after L1 (for MET type-1 use).
    """
    pt = pt_raw

    # L1FastJet(area, eta, pt, rho)
    c1 = _eval_numeric(refs.cL1, area, eta, pt, rho)
    pt = pt * c1
    pt_after_l1 = pt

    # L2Relative
    if has_phi_dependent_L2(year):
        c2 = _eval_numeric(refs.cL2, eta, phi, pt)
    else:
        c2 = _eval_numeric(refs.cL2, eta, pt)
    pt = pt * c2

    # Data residuals
    if isData:
        if requires_run_based_residual(year):
            run_for_res = representative_run_number(year, runs)
            cR = _eval_numeric(refs.cL2L3Res, run_for_res, eta, pt) 
        else:
            cR = _eval_numeric(refs.cL2L3Res, eta, pt) 
        pt = pt * cR

    return pt, pt_after_l1

# ---------------------------
# JES uncertainty shift (vectorized)
# ---------------------------
def jes_syst_shift(pt_nom, eta, refs_cs: correction.CorrectionSet, syst_name: str, var: str, print_evt: int):
    """
    syst_name node returns a fractional scale; apply 1 ± scale.
    """
    syst = refs_cs[syst_name]
    scale = _eval_numeric(syst, eta, pt_nom)
    debug_values("[Syst} Input jet pT:", pt_nom, indent=8, event_index=print_evt)
    if var == "Up":
        shift =  (1.0 + scale)
    else:
        shift =  (1.0 - scale)

    debug_values("[Syst} Shifted jet pT:", shift*pt_nom, indent=8, event_index=print_evt)
    return  shift*pt_nom

# ---------------------------
# JER smearing (vectorized; region-gated)
# ---------------------------
@dataclass
class JerBin:
    label: str
    etaMin: float
    etaMax: float
    ptMin: float
    ptMax: float

from ROOT import TRandom3
_rng = TRandom3(0)

def _eventwise_trandom3_z(runs, lumis, events, pt_in):
    """
    One standard-normal deviate per event using ROOT::TRandom3 with
    seed = event + run + lumi, then broadcast to all jets in that event.
    """
    # jets-per-event (jagged lengths)
    counts = ak.num(pt_in, axis=1)

    # Make sure we iterate per event (1D)
    runs_np   = ak.to_numpy(runs)
    lumis_np  = ak.to_numpy(lumis)
    events_np = ak.to_numpy(events)

    z_per_event = []
    for run, lumi, evt in zip(runs_np, lumis_np, events_np):
        _rng.SetSeed(int(evt) + int(run) + int(lumi))
        z_per_event.append(_rng.Gaus(0.0, 1.0))

    # Turn into an Awkward 1D array (length = n_events)
    z_ev = ak.Array(z_per_event)

    # --- EASIEST: broadcast to jagged jet-shape directly ---
    z, _ = ak.broadcast_arrays(z_ev, pt_in)
    return z


YEARS_WITH_PT = {"2022Pre", "2022Post", "2023Pre", "2023Post", "2024"}

def _jer_sf(refs, year, eta, pt, cat: str):
    if year in YEARS_WITH_PT:
        return _eval_with_string_last(refs.cJerSF, eta, pt, cat=cat)
    else:
        return _eval_with_string_last(refs.cJerSF, eta, cat=cat)

def jer_smear(pt_in, eta, phi, rho, year: str, refs: CorrectionRefs,
              var: str, runs, lumis, events, genIdx, gen_pt, gen_eta, gen_phi,
              mindr: float, region: Optional[JerBin], print_evt: int):

    debug_values("[JER} Input jet pT:", pt_in, indent=8, event_index=print_evt)
    # Resolution (same for all branches below)
    reso = _eval_numeric(refs.cReso, eta, pt_in, rho)

    # --- Per-jet region gating for SF ---
    if region is not None:
        aeta = abs(eta)
        in_reg = (aeta >= region.etaMin) & (aeta < region.etaMax) & \
                 (pt_in >= region.ptMin) & (pt_in < region.ptMax)

        sf_nom = _jer_sf(refs, year, eta, pt_in, cat="nom")
        sf_var = _jer_sf(refs, year, eta, pt_in, cat=var)
        sf = ak.where(in_reg, sf_var, sf_nom)
    else:
        sf = _jer_sf(refs, year, eta, pt_in, cat=var)

    # --- Gen match ---
    gpt, valid_idx = take_by_index(gen_pt,  genIdx, 0.0)
    geta, _        = take_by_index(gen_eta, genIdx, 0.0)
    gphi, _        = take_by_index(gen_phi, genIdx, 0.0)

    dphi = phi_mpi_pi(phi - gphi)
    dRmatch = np.sqrt((eta - geta) * (eta - geta) + dphi * dphi)

    is_match = valid_idx & (dRmatch < mindr) & (abs(pt_in - gpt) < 3.0 * reso * pt_in)
    debug_values("[JER] is_match :", is_match, indent=8, event_index=print_evt)

    corr_match   = np.maximum(0.0, 1.0 + (sf - 1.0) * (pt_in - gpt) / pt_in)
    debug_values("Matched JER corr :", corr_match, indent=8, event_index=print_evt)

    # ---------- Event-wide deterministic Gaussian (match C++ reseeding) ----------
    # C++ seed: event + run + lumiblock (same for every jet in the event)
    # Build one z per event and broadcast to jets.
    seed_ev = ak.values_astype(events, "uint64") \
              + ak.values_astype(runs,   "uint64") \
              + ak.values_astype(lumis,  "uint64")

    # Make a second seed deterministically from the first
    seed_ev_b = seed_ev ^ np.uint64(0x9E3779B97F4A7C15)

    z = _eventwise_trandom3_z(runs, lumis, events, pt_in)
    sigma = np.sqrt(np.maximum(sf * sf - 1.0, 0.0)) * reso
    corr_unmatch = np.maximum(0.0, 1.0 + z * sigma)

    corr_unmatch = np.maximum(0.0, 1.0 + z * sigma)
    debug_values("UnMatched JER corr :", corr_unmatch, indent=8, event_index=print_evt)

    corr = ak.where(is_match, corr_match, corr_unmatch)
    return pt_in * corr


# ---------------------------
# Systematic tag helpers (from your JSON layout)
# ---------------------------
# JES: map setName -> list[(fullTag, baseName)]
def get_syst_tag_names(year_json: dict, year: str) -> Dict[str, List[Tuple[str, str]]]:
    out = {}
    if year not in year_json or "ForUncertaintyJES" not in year_json[year]:
        return out
    for setName, arr in year_json[year]["ForUncertaintyJES"].items():
        pairs = []
        for item in arr:
            if isinstance(item, list) and len(item) >= 2:
                pairs.append((item[0], item[1]))
        out[setName] = pairs
    return out

# JER: map setName -> list[JerBin]
def get_jer_sets(year_json: dict, year: str) -> Dict[str, List[JerBin]]:
    out = {}
    if year not in year_json:
        return out
    j = year_json[year].get("ForUncertaintyJER", {})
    for setName, obj in j.items():
        bins = []
        for label, arr in obj.items():
            if isinstance(arr, list) and len(arr) == 4:
                bins.append(JerBin(label, arr[0], arr[1], arr[2], arr[3]))
        out[setName] = bins
    return out

# ---------------------------
# ROOT histogram helpers
# ---------------------------
def mkdirs(root_dir: ROOT.TDirectory, parts: List[str]) -> ROOT.TDirectory:
    cur = root_dir
    for p in parts:
        nxt = cur.Get(p)
        if not nxt:
            cur.cd()
            cur = cur.mkdir(p)
        else:
            cur = nxt
    cur.cd()
    return cur

def make_hists(fout: ROOT.TFile, year: str, isData: bool, era: Optional[str],
               syst_set: str, syst_name: str, also_nano: bool):
    parts = [year, "Data" if isData else "MC"]
    if isData and era:
        parts.append(era)
    parts += [syst_set, syst_name]

    # Preserve the current directory so we can restore it after creating the
    # histogram directory structure.  PyROOT updates gDirectory implicitly when
    # creating sub-directories.
    prev_dir = ROOT.gDirectory
    d = mkdirs(fout, parts)

    def h1(name):
        h = ROOT.TH1F(name, "", 50, 10.0, 510.0)
        h.Sumw2()
        h.SetDirectory(d)
        return h

    H = {
        "hJetPt_AK4": h1("hJetPt_AK4"),
        "hJetPt_AK8": h1("hJetPt_AK8"),
        "hMET": h1("hMET"),
        "hJetPt_AK4_Nano": h1("hJetPt_AK4_Nano") if also_nano else None,
        "hJetPt_AK8_Nano": h1("hJetPt_AK8_Nano") if also_nano else None,
        "hMET_Nano": h1("hMET_Nano") if also_nano else None,
    }

    if prev_dir:
        prev_dir.cd()

    return H, d


def write_histograms_to_directory(directory: ROOT.TDirectory, histograms: Dict[str, Optional[ROOT.TH1]]):
    """Write all non-null histograms to the provided directory."""

    if not isinstance(directory, ROOT.TDirectory):
        raise TypeError(
            f"write_histograms_to_directory: expected ROOT.TDirectory, got {type(directory)}"
        )

    prev_dir = ROOT.gDirectory
    directory.cd()
    try:
        for hist in histograms.values():
            if hist is None:
                continue
            # Ensure the histogram is owned by the destination directory and
            # write/overwrite it.  SetDirectory keeps ROOT from auto-assigning
            # it elsewhere if gDirectory changes in between calls.
            hist.SetDirectory(directory)
            hist.Write(hist.GetName(), ROOT.TObject.kOverwrite)
    finally:
        if prev_dir:
            prev_dir.cd()

def fill_h1_from_array(h: ROOT.TH1F, arr):
    if h is None:
        return
    flat = ak.to_numpy(ak.flatten(arr, axis=None))
    n_entries = float(flat.size)
    if flat.size == 0:
        h.SetEntries(n_entries)
        return
    counts, edges = np.histogram(flat, bins=h.GetNbinsX(),
                                 range=(h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax()))
    for i in range(1, h.GetNbinsX() + 1):
        h.SetBinContent(i, counts[i - 1])
        h.SetBinError(i, math.sqrt(counts[i - 1]))
    h.SetEntries(n_entries)

def _norm_dir_name(x, default="Nominal"):
    # treat None, "", "None" as default; always return a plain str
    if x is None: 
        return default
    s = str(x)
    return default if s.strip() == "" or s.strip().lower() == "none" else s

def ensure_dir(parent, name):
    import ROOT
    if not isinstance(parent, ROOT.TDirectory):
        raise TypeError(f"ensure_dir: 'parent' must be a TDirectory, got {type(parent)}")
    name = _norm_dir_name(name, default="Nominal")
    # Try GetDirectory; if not found, mkdir
    d = parent.GetDirectory(name)
    if not d:
        # ROOT sometimes caches; also try generic Get
        got = parent.Get(name)
        d = got if isinstance(got, ROOT.TDirectory) else parent.mkdir(name)
    d.cd()
    return d

def read_event_arrays(input_file: str, isData: bool):
    with uproot.open(input_file) as fin:
        t = fin["Events"]
        branches = [
            "run","luminosityBlock","event",
            "Rho_fixedGridRhoFastjetAll",
            "MET_pt","MET_phi",
            "RawMET_pt","RawMET_phi",
            "RawPuppiMET_pt","RawPuppiMET_phi",
            "nJet","Jet_pt","Jet_eta","Jet_phi","Jet_mass","Jet_rawFactor",
            "Jet_muonSubtrFactor","Jet_area","Jet_jetId","Jet_neEmEF","Jet_chEmEF",
            "nFatJet","FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass","FatJet_rawFactor","FatJet_area","FatJet_jetId",
        ]
        if not isData:  
            branches += [
                "Jet_genJetIdx","FatJet_genJetAK8Idx",
                "nGenJet","GenJet_pt","GenJet_eta","GenJet_phi",
                "nGenJetAK8","GenJetAK8_pt","GenJetAK8_eta","GenJetAK8_phi",
            ]
        return t.arrays(branches, library="ak", how=dict)

# ---------------------------
# Helper: apply a value only on selected positions, keep original elsewhere
# ---------------------------
def _only_nonoverlap(sel, updated, original):
    # sel, updated, original are jagged-with-identical layout
    return ak.where(sel, updated, original)

# ---------------------------
# Event processing (columnar)
# ---------------------------
def process_events(
    fout: ROOT.TFile,
    year: str,
    isData: bool,
    era: Optional[str],
    syst_kind: str,                 # "Nominal", "JES", "JER"
    jes_ak4_tag: Optional[str],
    jes_ak8_tag: Optional[str],
    jes_var: Optional[str],         # "Up" / "Down"
    jer_bin: Optional[JerBin],
    jer_var: Optional[str],         # "up" / "down" / "nom"
    refsAK4: CorrectionRefs,
    refsAK8: CorrectionRefs,
    jvm_ref,
    jvm_key: Optional[str],
    arrs,
    # NEW: allow overrides from tasks **kwargs
    syst_set: Optional[str] = None,
    syst_name: Optional[str] = None,
):
    # ---- debug header ----
    printDebug(f"[Debug] Processing {syst_kind} (year={year}, isData={isData}, era={era}, syst_set={syst_set}, syst_name={syst_name})")

    # -------- Decide histogram group (directory) and leaf (name) --------
    # 1) group / directory
    if syst_set is None:
        if   syst_kind == "Nominal": syst_set_name = "Nominal"
        elif syst_kind == "JES":     syst_set_name = "ForUncertaintyJES"
        elif syst_kind == "JER":     syst_set_name = "ForUncertaintyJER"
        else:                        syst_set_name = syst_kind
    else:
        syst_set_name = syst_set

    # 2) histogram name (leaf)
    if syst_name is None:
        if   syst_kind == "Nominal":
            syst_name_final = "Nominal"
        elif syst_kind == "JES":
            base = jes_ak4_tag or jes_ak8_tag or "JES"
            syst_name_final = f"{base}_{jes_var}" if jes_var else base
        elif syst_kind == "JER":
            if jer_bin is not None and jer_var in ("up", "down"):
                syst_name_final = f"{jer_bin.label}_{'Up' if jer_var=='up' else 'Down'}"
            else:
                syst_name_final = "JER"
        else:
            syst_name_final = syst_kind
    else:
        # tasks provided a base name; add suffixes where relevant
        if syst_kind == "JES" and jes_var:
            syst_name_final = f"{syst_name}_{jes_var}"
        elif syst_kind == "JER" and jer_bin is not None and jer_var in ("up", "down"):
            syst_name_final = f"{syst_name}_{'Up' if jer_var=='up' else 'Down'}"
        else:
            syst_name_final = syst_name

    # ---- proceed ----
    # Aliases
    run  = arrs["run"]
    lumi = arrs["luminosityBlock"]
    evt  = arrs["event"]
    rho  = arrs["Rho_fixedGridRhoFastjetAll"]

    # Hist group
    H, hist_dir = make_hists(
        fout,
        year,
        isData,
        era,
        syst_set_name,
        syst_name_final,
        also_nano=(syst_kind == "Nominal"),
    )

    printDebug(f"[Debug] make_hists -> set='{syst_set_name}', name='{syst_name_final}'")

    # --- AK8 preselect
    ak8_pt = arrs["FatJet_pt"]
    ak8_eta = arrs["FatJet_eta"]
    ak8_phi = arrs["FatJet_phi"]
    ak8_area = arrs["FatJet_area"]
    ak8_rawF = arrs["FatJet_rawFactor"]
    ak8_pt_raw = ak8_pt * (1.0 - ak8_rawF)

    ak8_sel = (ak8_pt >= 100.0) & (abs(ak8_eta) <= 5.2)
    ak8_idx = ak.local_index(ak8_pt, axis=1)
    ak8_keep_idx = ak8_idx[ak8_sel]

    # For Nano hist (only in nominal)
    if H["hJetPt_AK8_Nano"] is not None:
        fill_h1_from_array(H["hJetPt_AK8_Nano"], ak8_pt[ak8_sel])

    # --- AK4 preselect (non-overlapping with selected AK8, ΔR>0.6)
    ak4_pt = arrs["Jet_pt"]
    ak4_eta = arrs["Jet_eta"]
    ak4_phi = arrs["Jet_phi"]
    ak4_area = arrs["Jet_area"]
    ak4_rawF = arrs["Jet_rawFactor"]
    ak4_muSub = arrs["Jet_muonSubtrFactor"]
    ak4_neEm = arrs["Jet_neEmEF"]
    ak4_chEm = arrs["Jet_chEmEF"]
    ak4_id = arrs["Jet_jetId"]
    ak4_pt_raw = ak4_pt * (1.0 - ak4_rawF)

    basic_ak4 = (ak4_pt >= 15.0) & (abs(ak4_eta) <= 5.2)

    # --- AK4–AK8 overlap without `vector`, robust for empty AK8 events ---

    # Selected AK8 (per event, jagged)
    ak8_eta_sel = ak8_eta[ak8_sel]
    ak8_phi_sel = ak8_phi[ak8_sel]

    # Pairwise differences with explicit new axes:
    # shapes: (ev, nAK4, 1)  minus  (ev, 1, nAK8_sel)  -> (ev, nAK4, nAK8_sel)
    deta = ak4_eta[:, :, None] - ak8_eta_sel[:, None, :]
    dphi = phi_mpi_pi(ak4_phi[:, :, None] - ak8_phi_sel[:, None, :])

    # ΔR^2 and min over the AK8 dimension. If an event has no AK8, the third
    # dimension is length 0; `initial` makes the min a large number.
    dr2 = deta * deta + dphi * dphi
    min_dr = np.sqrt(ak.min(dr2, axis=2, initial=1.0e12))

    # Overlap mask and final AK4 selection
    overlap = (min_dr < 0.6)
    overlap = ak.fill_none(overlap, False)  # paranoia guard
    ak4_sel_nonoverlap = basic_ak4 & (~overlap)

    # Nano AK4 hist
    if H["hJetPt_AK4_Nano"] is not None:
        fill_h1_from_array(H["hJetPt_AK4_Nano"], ak4_pt[ak4_sel_nonoverlap])

    # Nano MET hist
    if H["hMET_Nano"] is not None:
        fill_h1_from_array(H["hMET_Nano"], arrs["MET_pt"])

    if applyOnlyOnAK4 or applyOnlyOnAK8:
        applyOnAK4AndAK8 = False
    print_evt = None
    if isDebug:
        print_evt = _find_debug_event_index(
            ak4_sel_nonoverlap,
            ak8_sel,
            applyOnlyOnAK4,
            applyOnlyOnAK8,
            applyOnAK4AndAK8,
        )
        if print_evt is not None:
            def _count_selected(selection) -> int:
                if selection is None or len(selection) <= print_evt:
                    return 0
                try:
                    return int(ak.sum(selection[print_evt]))
                except Exception:
                    return 0

            num_ak4 = _count_selected(ak4_sel_nonoverlap)
            num_ak8 = _count_selected(ak8_sel)
            printDebug(
                f"Debugging event index {print_evt}:selected AK8 jets = {num_ak8}, selected AK4 jets = {num_ak4}",
                indent=2,
            )
        else:
            printDebug(
                "No event found with the requested jet selection for debug output.",
                indent=2,
            )

    print_ak4_debug = print_evt is not None and (applyOnlyOnAK4 or applyOnAK4AndAK8)
    print_ak8_debug = print_evt is not None and (applyOnlyOnAK8 or applyOnAK4AndAK8)

    # ---------------------------
    # JES nominal for AK4/AK8 (always applied first)
    # ---------------------------

    # --- AK4: compute on all, commit only on non-overlapping
    if print_ak4_debug:
        debug_values("[AK4] NanoAOD pt:", ak4_pt, indent=4, event_index=print_evt, mask=ak4_sel_nonoverlap)
        debug_values("[AK4] Raw (undo rawFactor) pt:", ak4_pt_raw, indent=6, event_index=print_evt, mask=ak4_sel_nonoverlap)

    # Compute for all AK4 (shape-stable)
    ak4_pt_corr_nom_all, ak4_pt_after_l1_all = jes_nominal(
        pt_raw=ak4_pt_raw, eta=ak4_eta, phi=ak4_phi, area=ak4_area, rho=rho,
        year=year, isData=isData, runs=run, refs=refsAK4
    )

    # Commit only for non-overlapping AK4; overlapped keep original Nano pt (or choose ak4_pt_raw if you prefer)
    ak4_pt_after_l1 = _only_nonoverlap(ak4_sel_nonoverlap, ak4_pt_after_l1_all, ak4_pt_raw)
    ak4_pt_corr_nom = _only_nonoverlap(ak4_sel_nonoverlap, ak4_pt_corr_nom_all, ak4_pt)

    if print_ak4_debug:
        debug_values("[AK4] After L1FastJet pt (non-overlap only):",
                     ak4_pt_after_l1, indent=6, event_index=print_evt, mask=ak4_sel_nonoverlap)
        debug_values("[AK4] JES nominal corrected pt (non-overlap only):",
                     ak4_pt_corr_nom, indent=6, event_index=print_evt, mask=ak4_sel_nonoverlap)

    # --- AK8: unchanged
    if print_ak8_debug:
        debug_values("[AK8] NanoAOD pt:", ak8_pt, indent=4, event_index=print_evt, mask=ak8_sel)
        debug_values("[AK8] Raw (undo rawFactor) pt:", ak8_pt_raw, indent=6, event_index=print_evt, mask=ak8_sel)

    ak8_pt_corr, ak8_pt_after_l1 = jes_nominal(
        pt_raw=ak8_pt_raw, eta=ak8_eta, phi=ak8_phi, area=ak8_area, rho=rho,
        year=year, isData=isData, runs=run, refs=refsAK8
    )
    if print_ak8_debug:
        debug_values("[AK8] After L1FastJet pt:", ak8_pt_after_l1, indent=6, event_index=print_evt, mask=ak8_sel)
        debug_values("[AK8] JES nominal corrected pt:", ak8_pt_corr, indent=6, event_index=print_evt, mask=ak8_sel)

    # ---------------------------
    # JES systematics (MC only) if requested
    # ---------------------------
    if is_mc(isData) and syst_kind == "JES":
        # Shift full arrays, then commit only on non-overlap for AK4
        if (applyOnlyOnAK4 or applyOnAK4AndAK8) and jes_ak4_tag:
            ak4_pt_corr_nom_all = jes_syst_shift(
                ak4_pt_corr_nom_all, ak4_eta, refsAK4.cs, jes_ak4_tag, jes_var, print_evt
            )
            ak4_pt_corr_nom = _only_nonoverlap(ak4_sel_nonoverlap, ak4_pt_corr_nom_all, ak4_pt)

        if (applyOnlyOnAK8 or applyOnAK4AndAK8) and jes_ak8_tag:
            ak8_pt_corr = jes_syst_shift(ak8_pt_corr, ak8_eta, refsAK8.cs, jes_ak8_tag, jes_var, print_evt)

    # ---------------------------
    # JER (MC only): nominal or region-gated up/down
    # ---------------------------
    if is_mc(isData):
        jet_gen_idx = arrs["Jet_genJetIdx"]
        gen_pt  = arrs["GenJet_pt"]
        gen_eta = arrs["GenJet_eta"]
        gen_phi = arrs["GenJet_phi"]

        fat_gen_idx = arrs["FatJet_genJetAK8Idx"]
        gen8_pt  = arrs["GenJetAK8_pt"]
        gen8_eta = arrs["GenJetAK8_eta"]
        gen8_phi = arrs["GenJetAK8_phi"]

        jer_var_eff = jer_var if syst_kind == "JER" else "nom"

        # --- AK4: smear full array, then commit only on non-overlap
        if (applyOnlyOnAK4 or applyOnAK4AndAK8):
            ak4_pt_corr_all = jer_smear(
                pt_in=ak4_pt_corr_nom_all, eta=ak4_eta, phi=ak4_phi, rho=rho, year=year, refs=refsAK4,
                var=jer_var_eff, runs=run, lumis=lumi, events=evt,
                genIdx=jet_gen_idx, gen_pt=gen_pt, gen_eta=gen_eta, gen_phi=gen_phi,
                mindr=0.2, region=jer_bin, print_evt=print_evt
            )
        else:
            ak4_pt_corr_all = ak4_pt_corr_nom_all

        # Commit only for non-overlap; overlapped keep original Nano pt
        ak4_pt_corr = _only_nonoverlap(ak4_sel_nonoverlap, ak4_pt_corr_all, ak4_pt)

        # --- AK8 unchanged
        if (applyOnlyOnAK8 or applyOnAK4AndAK8):
            ak8_pt_corr = jer_smear(
                pt_in=ak8_pt_corr, eta=ak8_eta, phi=ak8_phi, rho=rho, year=year, refs=refsAK8,
                var=jer_var_eff, runs=run, lumis=lumi, events=evt,
                genIdx=fat_gen_idx, gen_pt=gen8_pt, gen_eta=gen8_eta, gen_phi=gen8_phi,
                mindr=0.4, region=jer_bin, print_evt=print_evt
            )
    else:
        # Data: no JER; still enforce non-overlap commit policy
        ak4_pt_corr = _only_nonoverlap(ak4_sel_nonoverlap, ak4_pt_corr_nom_all, ak4_pt)

    # Debug prints (unchanged masks)
    if print_ak4_debug:
        debug_values("[AK4] Final (JER) corrected pt sample (non-overlap only):",
                     ak4_pt_corr, indent=4, event_index=print_evt, mask=ak4_sel_nonoverlap)
    if print_ak8_debug:
        debug_values("[AK8] Final (JER) corrected pt:", ak8_pt_corr, indent=4, event_index=print_evt, mask=ak8_sel)


    # ---------------------------
    # MET Type-1 propagation (AK4 only)
    # ---------------------------
    if applyOnMET:
        # Nano "raw" MET choice per year
        met_raw_pt = arrs["RawPuppiMET_pt"] if uses_puppi_met(year) else arrs["RawMET_pt"]
        met_raw_phi = arrs["RawPuppiMET_phi"] if uses_puppi_met(year) else arrs["RawMET_phi"]
        met_raw_px = met_raw_pt * np.cos(met_raw_phi)
        met_raw_py = met_raw_pt * np.sin(met_raw_phi)
        
        debug_values("[MET] Nano MET_pt :", arrs["MET_pt"], indent=4, event_index=print_evt)
        debug_values("[MET] Raw MET pt :", met_raw_pt, indent=4, event_index=print_evt)

        debug_values("[MET] AK4 Jet raw pt:", ak4_pt_raw, indent=6, event_index=print_evt)
        pt_raw_mu = ak4_pt_raw * (1.0 - ak4_muSub)
        debug_values("[MET] AK4 Jet raw pt after muon subtraction:", pt_raw_mu, indent=6, event_index=print_evt)

        # L1FastJet(area, eta, pt, rho)
        l1_corr = _eval_numeric(refsAK4.cL1, ak4_area, ak4_eta, pt_raw_mu, rho)
        pt_l1 = pt_raw_mu * l1_corr
        debug_values("[MET] AK4 Jet pt after L1FastJet:", pt_l1, indent=8, event_index=print_evt)

        # L2Relative
        if has_phi_dependent_L2(year):
            l2_corr = _eval_numeric(refsAK4.cL2, ak4_eta, ak4_phi, pt_l1)
        else:
            l2_corr = _eval_numeric(refsAK4.cL2, ak4_eta, pt_l1)
        pt_tmp = pt_l1 * l2_corr
        debug_values("[MET] AK4 Jet pt after L2Relative:", pt_tmp, indent=8, event_index=print_evt)

        # Residuals (data only)
        if isData:
            if requires_run_based_residual(year):
                run_for_res = representative_run_number(year, run)
                res_corr = _eval_numeric(refsAK4.cL2L3Res, run_for_res, ak4_eta, pt_tmp)
            else:
                res_corr = _eval_numeric(refsAK4.cL2L3Res, ak4_eta, pt_tmp)
            pt_tmp = pt_tmp * res_corr
            debug_values("[MET] AK4 Jet pt after residual corrections:", pt_tmp, indent=8, event_index=print_evt)

        # JES syst for MET path (MC only, only on this pass)
        if is_mc(isData) and syst_kind == "JES" and jes_ak4_tag:
            cnode = refsAK4.cs[jes_ak4_tag]
            scale = _eval_numeric(cnode, ak4_eta, pt_tmp)
            shift = (1.0 + scale) if jes_var == "Up" else (1.0 - scale)
            pt_tmp = pt_tmp * shift
            debug_values("[MET] AK4 Jet pt after JES systematic:", pt_tmp, indent=8, event_index=print_evt)

        # JER for MET path (MC only)
        if is_mc(isData):
            if "Jet_genJetIdx" in arrs:
                gen_idx = arrs["Jet_genJetIdx"]
                gpt = arrs["GenJet_pt"]; geta = arrs["GenJet_eta"]; gphi = arrs["GenJet_phi"]
            else:
                gen_idx = ak.zeros_like(ak4_pt, dtype=np.int32) - 1
                gpt = ak.Array([[]]) * 0.0; geta = gpt; gphi = gpt

            jer_var_eff = jer_var if syst_kind == "JER" else "nom"
            pt_corr_met = jer_smear(
                pt_in=pt_tmp, eta=ak4_eta, phi=ak4_phi, rho=rho, year=year, refs=refsAK4,
                var=jer_var_eff, runs=run, lumis=lumi, events=evt,
                genIdx=gen_idx, gen_pt=gpt, gen_eta=geta, gen_phi=gphi,
                mindr=0.2, region=jer_bin, print_evt=print_evt
            )
            debug_values("[MET] AK4 Jet pt after JER smearing:", pt_corr_met, indent=8, event_index=print_evt)
        else:
            pt_corr_met = pt_tmp
            debug_values("[MET] AK4 Jet pt for MET propagation:", pt_corr_met, indent=8, event_index=print_evt)

        pass_for_met = (pt_corr_met > 15.0) & (abs(ak4_eta) < 5.2) & ((ak4_chEm + ak4_neEm) < 0.9)
        dpt = ak.where(pass_for_met, pt_corr_met - pt_l1, 0.0)
        sum_px = ak.sum(dpt * np.cos(ak4_phi), axis=1)
        sum_py = ak.sum(dpt * np.sin(ak4_phi), axis=1)

        met_px = met_raw_px - sum_px
        met_py = met_raw_py - sum_py
        met_pt_corr = np.hypot(met_px, met_py)
        debug_values("[MET] Type-1 corrected MET pt:", met_pt_corr, indent=4, event_index=print_evt)
        fill_h1_from_array(H["hMET"], met_pt_corr)
    else:
        debug_values("[MET] MET_pt without Type-1 corrections:", arrs["MET_pt"], indent=4, event_index=print_evt)


    # ---------------------------
    # Jet Veto Map veto (count only; no event filtering for hists in this port)
    # (In your C++ you 'continue' after veto; here we replicate the count
    #  and DO NOT fill more objects since we already filled above.)
    # ---------------------------
    use_jvm = jvm_ref is not None and jvm_key is not None
    if use_jvm:
        msel = (abs(ak4_eta) < 5.191) & (abs(ak4_phi) <= np.pi) & (ak4_id >= 6) & \
               (ak4_pt >= 15.0) & ((ak4_chEm + ak4_neEm) <= 0.9)
        jvm_vals = _eval_with_string_first(jvm_ref, ak4_eta[msel], ak4_phi[msel], cat=jvm_key)
        flagged = ak.unflatten(ak.to_numpy(ak.flatten(jvm_vals > 0.0, axis=None)),
                               ak.num(ak4_eta[msel]))
        vetoed = ak.any(flagged, axis=1)
        print(f"   Number of events vetoed due to JetVetoMap: {int(ak.sum(vetoed))}")

    # ---------------------------
    # Fill jet pT hists (post-corrections) with the selected sets
    # ---------------------------
    if applyOnlyOnAK8 or applyOnAK4AndAK8:
        fill_h1_from_array(H["hJetPt_AK8"], ak8_pt_corr[ak8_sel])
    if applyOnlyOnAK4 or applyOnAK4AndAK8:
        fill_h1_from_array(H["hJetPt_AK4"], ak4_pt_corr[ak4_sel_nonoverlap])

    # Persist histograms to the output file inside the correct directory tree.
    write_histograms_to_directory(hist_dir, H)

# ---------------------------
# High-level driver (years & systematics)
# ---------------------------
def process_with_nominal_and_syst(input_file: str, fout: ROOT.TFile,
                                  year: str, isData: bool, era: Optional[str] = None):
    # Tag JSONs (cached)
    cfgAK4 = CorrCache.get_cfg("AK4")
    cfgAK8 = CorrCache.get_cfg("AK8")

    # JES sets for AK4/AK8 (derive once per year)
    sAK4 = get_syst_tag_names(cfgAK4, year)
    sAK8 = get_syst_tag_names(cfgAK8, year) if year in cfgAK8 else sAK4  # fallback

    # JER sets (use AK4 JSON for binning)
    jerSets = get_jer_sets(cfgAK4, year)

    # Build corrections/JVM once per (year,isData,era)
    refsAK4 = CorrCache.get_refs("AK4", year, isData, era)
    refsAK8 = CorrCache.get_refs("AK8", year, isData, era)
    jvm_ref, jvm_key = CorrCache.get_jvm(year)

    # --- NEW: fresh arrays provider so every pass starts from raw Nano
    def fresh_arrays():
        return read_event_arrays(input_file, isData)

    tasks: List[Tuple[str, Dict[str, object]]] = []

    tasks.append((
        " [Nominal]",
        {
            "syst_kind": "Nominal",
            "jes_ak4_tag": None,
            "jes_ak8_tag": None,
            "jes_var": None,
            "jer_bin": None,
            "jer_var": None,
            "syst_set":"Nominal",
            "syst_name":"Nominal",
        },
    ))

    if is_mc(isData):
        # 1) Correlated JES systematics
        if applyOnlyOnAK4:
            for setName, pairs in sAK4.items():
                for fullTag, baseName in pairs:
                    for var in ("Up", "Down"):
                        tasks.append((
                            f"\n [JES Syst]: {baseName}_{var}",
                            {
                                "syst_kind": "JES",
                                "jes_ak4_tag": fullTag,
                                "jes_ak8_tag": None,
                                "jes_var": var,
                                "jer_bin": None,
                                "jer_var": None,
                                "syst_set": setName,
                                "syst_name": baseName
                            },
                        ))
        elif applyOnlyOnAK8:
            for setName, pairs in sAK8.items():
                for fullTag, baseName in pairs:
                    for var in ("Up", "Down"):
                        tasks.append((
                            f"\n [JES Syst]: {baseName}_{var}",
                            {
                                "syst_kind": "JES",
                                "jes_ak4_tag": None,
                                "jes_ak8_tag": fullTag,
                                "jes_var": var,
                                "jer_bin": None,
                                "jer_var": None,
                                "syst_set": setName,
                                "syst_name": baseName
                            },
                        ))
        else:
            # loop over every set present in either AK4 or AK8
            for setName in sorted(set(sAK4.keys()) | set(sAK8.keys())):
                ak4_pairs = sAK4.get(setName, [])
                ak8_pairs = sAK8.get(setName, [])
            
                # base -> full tag maps for this set
                base_to_ak4 = {base: full for (full, base) in ak4_pairs}
                base_to_ak8 = {base: full for (full, base) in ak8_pairs}
            
                # intersection per-set
                common_bases = sorted(set(base_to_ak4) & set(base_to_ak8))
                for base in common_bases:
                    for var in ("Up", "Down"):
                        tasks.append((
                            f"\n [JES Syst]: {base}_{var}",
                            {
                                "syst_kind": "JES",
                                "jes_ak4_tag": base_to_ak4[base],
                                "jes_ak8_tag": base_to_ak8[base],
                                "jes_var": var,
                                "jer_bin": None,
                                "jer_var": None,
                                "syst_set": setName,   
                                "syst_name": base
                            },
                        ))
            

        # 2) JER systematics (region-gated up/down)
        for setName, bins in jerSets.items():
            for b in bins:
                for var in ("up", "down"):
                    tasks.append((
                        f"\n [JER Syst]: {setName}/{b.label}_{'Up' if var == 'up' else 'Down'}",
                        {
                            "syst_kind": "JER",
                            "jes_ak4_tag": None,
                            "jes_ak8_tag": None,
                            "jes_var": None,
                            "jer_bin": b,
                            "jer_var": var,
                            "syst_set": setName,   
                            "syst_name": b.label
                        },
                    ))

    desc_parts = ["Data" if isData else "MC", year]
    if isData and era:
        desc_parts.append(era)
    progress_desc = " | ".join(desc_parts)
    progress = tqdm(total=len(tasks), desc=f"Passes {progress_desc}", unit="pass") if _TQDM_AVAILABLE else None

    for label, kwargs in tasks:
        _progress_print(str(label))
        process_events(
            fout=fout,
            year=year,
            isData=isData,
            era=era,
            refsAK4=refsAK4,
            refsAK8=refsAK8,
            jvm_ref=jvm_ref,
            jvm_key=jvm_key,
            arrs=fresh_arrays(),
            **kwargs,
        )
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()



# ---------------------------
# Main (mirrors your macro)
# ---------------------------
def main():
    # Input files (same names as C++)
    fInputData = "NanoAOD_Data.root"
    fInputMc   = "NanoAOD_MC.root"

    # Output file
    outName = "output_Py.root"
    fout = ROOT.TFile(outName, "RECREATE")

    # Year configs (match C++)
    mcYears = [
        "2016Pre",
        #"2016Post","2017","2018",
        #"2022Pre","2022Post","2023Pre","2023Post","2024"
    ]
    dataConfigs = [
        ("2016Pre","Era2016PreBCD"),
        #("2016Pre","Era2016PreEF"),
        #("2016Post","Era2016PostFGH"),
        #("2017","Era2017B"),("2017","Era2017C"),("2017","Era2017D"),
        #("2017","Era2017E"),("2017","Era2017F"),
        #("2018","Era2018A"),("2018","Era2018B"),("2018","Era2018C"),("2018","Era2018D"),
        #("2022Pre","Era2022C"),("2022Pre","Era2022D"),
        #("2022Post","Era2022E"),("2022Post","Era2022F"),("2022Post","Era2022G"),
        #("2023Pre","Era2023PreAll"),
        #("2023Post","Era2023PostAll"),
        #("2024","Era2024All"),
    ]

    # MC
    for year in mcYears:
        print("-----------------")
        print(f"[MC] : {year}")
        print("-----------------")
        process_with_nominal_and_syst(fInputMc, fout, year, isData=False)

    # Data
    for year, era in dataConfigs:
        print("-----------------")
        print(f"[Data] : {year} : {era}")
        print("-----------------")
        process_with_nominal_and_syst(fInputData, fout, year, isData=True, era=era)

    fout.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    print(f"Wrote output to {outName}")

if __name__ == "__main__":
    main()

