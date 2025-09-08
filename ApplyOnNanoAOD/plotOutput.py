#!/usr/bin/env python3
import argparse, json, os, sys
import numpy as np
import uproot
import matplotlib.pyplot as plt

EPS = 1e-12

def to_numpy_norm(hobj):
    """Return (y, edges) for TH1; normalize to unit area (shape)."""
    y, edges = hobj.to_numpy(flow=False)
    integral = np.sum(y)
    if integral > 0:
        y = y / integral
    return y.astype(float), edges.astype(float)

def read_hist(tree, path):
    """Return (y, edges) or (None, None) if missing."""
    try:
        h = tree[path]
    except KeyError:
        return None, None
    try:
        return to_numpy_norm(h)
    except Exception:
        return None, None

def centers(edges):
    return 0.5*(edges[:-1] + edges[1:])

def get_jes_names(cfg_year, subset_key):
    # JES lists are [["tagName", "CMS_label"], ...]
    lst = cfg_year["ForUncertaintyJES"][subset_key]
    return [pair[1] for pair in lst]

def get_jer_names(cfg_year, subset_key):
    # JER dicts are {"CMS_label": [etaMin, etaMax, ptMin, ptMax], ...}
    dct = cfg_year["ForUncertaintyJER"][subset_key]
    return list(dct.keys())

def per_set_nuisance_names(cfg_year, set_key):
    # Map set key -> list of nuisance directory names (no _Up/_Down)
    if "JES" in set_key:
        if set_key not in cfg_year["ForUncertaintyJES"]:
            return []
        return get_jes_names(cfg_year, set_key)
    elif "JER" in set_key:
        if set_key not in cfg_year["ForUncertaintyJER"]:
            return []
        return get_jer_names(cfg_year, set_key)
    else:
        return []

def total_rel_band(file_handle, year, hist_name, set_dir, nuisances, nom_y):
    """
    Compute bin-wise total relative uncertainty for a given set by
    adding max(|up-nom|,|down-nom|)/nom in quadrature across nuisances.
    """
    if len(nuisances) == 0:
        return np.zeros_like(nom_y)

    denom = np.maximum(nom_y, EPS)
    sumsq = np.zeros_like(nom_y, dtype=float)

    base = f"{year}/MC/{set_dir}"
    for nu in nuisances:
        up_path   = f"{base}/{nu}_Up/{hist_name}"
        down_path = f"{base}/{nu}_Down/{hist_name}"

        up_y, _   = read_hist(file_handle, up_path)
        dn_y, _   = read_hist(file_handle, down_path)

        # If a variation is missing, treat its delta as 0 for that side
        diff_up = np.abs(up_y - nom_y) if up_y is not None else 0.0
        diff_dn = np.abs(dn_y - nom_y) if dn_y is not None else 0.0

        delta = np.maximum(diff_up, diff_dn) / denom
        # nan-safe (e.g. if both up & down are None)
        delta = np.nan_to_num(delta, copy=False)
        sumsq += delta**2

    return np.sqrt(sumsq)

def plot_one_year(file_handle, cfg_year, year, era, hist_name, outdir, show_sets):
    # --- Nominal MC ---
    mc_nom_path = f"{year}/MC/Nominal/Nominal/{hist_name}"
    mc_nom_y, e = read_hist(file_handle, mc_nom_path)
    if mc_nom_y is None:
        print(f"[{year}] MISSING MC nominal: {mc_nom_path}")
        return
    x = centers(e)

    # --- Nominal Data for the chosen era ---
    data_nom_path = f"{year}/Data/{era}/Nominal/Nominal/{hist_name}"
    data_y, eD = read_hist(file_handle, data_nom_path)
    if data_y is None:
        print(f"[{year}] MISSING Data nominal: {data_nom_path}")
        return
    if not np.allclose(e, eD):
        print(f"[{year}] WARNING: Data/MC binning differs; proceeding with MC binning.")

    # --- Build total bands for each requested set ---
    set_to_dir = {
        "ForUncertaintyJESReduced": "ForUncertaintyJESReduced",
        "ForUncertaintyJESTotal":   "ForUncertaintyJESTotal",
        "ForUncertaintyJESFull":    "ForUncertaintyJESFull",
        "ForUncertaintyJERTotal":   "ForUncertaintyJERTotal",
        "ForUncertaintyJERFull":    "ForUncertaintyJERFull",
    }

    bands = []  # list of (label, y_low, y_high)
    for set_key in show_sets:
        if "JES" in set_key and "ForUncertaintyJES" not in cfg_year:
            continue
        if "JER" in set_key and "ForUncertaintyJER" not in cfg_year:
            continue

        nuisances = per_set_nuisance_names(cfg_year, set_key)
        set_dir   = set_to_dir[set_key]

        rel = total_rel_band(file_handle, year, hist_name, set_dir, nuisances, mc_nom_y)
        y_low  = mc_nom_y * (1.0 - rel)
        y_high = mc_nom_y * (1.0 + rel)
        bands.append((set_key.replace("ForUncertainty", "").replace("JES", "JES ")
                      .replace("JER", "JER "), y_low, y_high))

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    # MC nominal (step)
    plt.step(e[:-1], mc_nom_y, where="post", linewidth=1.8, label="MC (Nominal)")

    # Data (markers). Poisson error (on counts) divided by total counts ~ sqrt(y*I)/I = sqrt(y/I)
    # But we normalized; for shape-only overlay it's common to show markers without yerr.
    plt.errorbar(x, data_y, fmt="o", ms=3.5, capsize=0, label=f"Data {era}")

    # Bands (draw darker later ones on top)
    # Order: Reduced, JESTotal, JESFull, JERTotal, JERFull (if present)
    order = {name: i for i, name in enumerate(
        ["JES Reduced", "JES Total", "JES Full", "JER Total", "JER Full"]
    )}
    bands.sort(key=lambda t: order.get(t[0], 99))

    for lbl, lo, hi in bands:
        plt.fill_between(x, lo, hi, step="mid", alpha=0.25, linewidth=0.0, label=lbl)

    plt.title(f"{hist_name} — {year} / {era} (shape-normalized)")
    plt.xlabel("pT [GeV]" if "JetPt" in hist_name else hist_name)
    plt.ylabel("Normalized entries")
    plt.ylim(bottom=0)
    plt.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    base = f"{hist_name}_{year}_{era}".replace("/", "_")
    out_png = os.path.join(outdir, base + ".png")
    out_pdf = os.path.join(outdir, base + ".pdf")
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved: {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Overlay Data vs MC with JES/JER bands (shape-only).")
    ap.add_argument("--config", required=True, help="Path to JSON config (your JERC map).")
    ap.add_argument("--root",   required=True, help="Path to output.root produced by your job.")
    ap.add_argument("--era",    required=True, help="Data era name, e.g. Era2016PreBCD.")
    ap.add_argument("--hist",   default="hJetPt_AK4", choices=["hJetPt_AK4","hJetPt_AK8","hMET"],
                    help="Histogram name to plot (must exist in the ROOT file).")
    ap.add_argument("--sets",   nargs="+",
                    default=["ForUncertaintyJESReduced","ForUncertaintyJESTotal","ForUncertaintyJESFull",
                             "ForUncertaintyJERTotal","ForUncertaintyJERFull"],
                    help="Which sets to overlay (any subset of the defaults).")
    ap.add_argument("--years",  nargs="*", default=None,
                    help="Limit to these years; default = all years in config.")
    ap.add_argument("--outdir", default="plots_jerc_bands", help="Where to write figures.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    years = args.years if args.years else list(cfg.keys())
    missing = []

    with uproot.open(args.root) as rf:
        for year in years:
            if year not in cfg:
                print(f"Config has no year '{year}', skipping.")
                continue

            # If the chosen era is not listed in config['data'], still try if the tree exists.
            cfg_year = cfg[year]
            era_keys = set(cfg_year.get("data", {}).keys())
            if len(era_keys) and (args.era not in era_keys):
                print(f"[{year}] Era '{args.era}' not in config[data]; attempting anyway if present in ROOT.")

            # Quick existence check in ROOT:
            mc_nom_path = f"{year}/MC/Nominal/Nominal/{args.hist}"
            data_nom_path = f"{year}/Data/{args.era}/Nominal/Nominal/{args.hist}"
            if mc_nom_path not in rf:
                missing.append(mc_nom_path);  continue
            if data_nom_path not in rf:
                missing.append(data_nom_path); continue

            plot_one_year(rf, cfg_year, year, args.era, args.hist, args.outdir, args.sets)

    if missing:
        print("\nMissing objects/paths:")
        for m in missing:
            print("  -", m)

if __name__ == "__main__":
    main()

