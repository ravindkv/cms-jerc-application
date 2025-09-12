#!/usr/bin/env python3
import argparse, json, os, sys
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

EPS = 1e-12
# Consistent colors for uncertainty sets (top bands and bottom ratio lines)
# Stable colors for each uncertainty set
UNC_COLORS = {
    "JES Reduced": "C2",
    "JES Total":   "C3",
    "JES Full":    "C4",
    "JER Total":   "C5",
    "JER Full":    "C6",
}

# Helpers: consistent labels + color lookup with safe fallback
_COLOR_CACHE = {}
def _norm_unc_label(set_key: str) -> str:
    lbl = set_key.replace("ForUncertainty", "")
    lbl = lbl.replace("JES", "JES ").replace("JER", "JER ")
    return " ".join(lbl.split())  # collapse double spaces just in case

def _color_for(lbl: str) -> str:
    if lbl in UNC_COLORS:
        return UNC_COLORS[lbl]
    # fallback: stable pick from matplotlib cycle (cached)
    if lbl not in _COLOR_CACHE:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _COLOR_CACHE[lbl] = cycle[len(_COLOR_CACHE) % len(cycle)]
    return _COLOR_CACHE[lbl]



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

def plot_one_year(file_handle, cfg_year, year, era, hist_name, pdf, show_sets):
    # --- Nominal MC ---
    mc_nom_path = f"{year}/MC/Nominal/Nominal/{hist_name}"
    mc_nom_y, e = read_hist(file_handle, mc_nom_path)
    if mc_nom_y is None:
        print(f"[{year}] MISSING MC nominal: {mc_nom_path}")
        return
    x = centers(e)

    mc_nom_nano_path = f"{year}/MC/Nominal/Nominal/{hist_name}_Nano"
    mc_nom_nano_y, e_nano = read_hist(file_handle, mc_nom_nano_path)

    # --- Nominal Data for the chosen era ---
    data_nom_path = f"{year}/Data/{era}/Nominal/Nominal/{hist_name}"
    data_y, eD = read_hist(file_handle, data_nom_path)
    if data_y is None:
        print(f"[{year}] MISSING Data nominal: {data_nom_path}")
        return
    if not np.allclose(e, eD):
        print(f"[{year}] WARNING: Data/MC binning differs; proceeding with MC binning.")

    data_nom_nano_path = f"{year}/Data/{era}/Nominal/Nominal/{hist_name}_Nano"
    data_y_nano, eD_nano = read_hist(file_handle, data_nom_nano_path)

    # --- Build total bands for each requested set ---
    set_to_dir = {
        "ForUncertaintyJESReduced": "ForUncertaintyJESReduced",
        "ForUncertaintyJESTotal":   "ForUncertaintyJESTotal",
        "ForUncertaintyJESFull":    "ForUncertaintyJESFull",
        "ForUncertaintyJERTotal":   "ForUncertaintyJERTotal",
        "ForUncertaintyJERFull":    "ForUncertaintyJERFull",
    }

    # bands: list of (label, y_low, y_high, rel)
    bands = []
    for set_key in show_sets:
        if "JES" in set_key and "ForUncertaintyJES" not in cfg_year:
            continue
        if "JER" in set_key and "ForUncertaintyJER" not in cfg_year:
            continue

        nuisances = per_set_nuisance_names(cfg_year, set_key)
        set_dir   = set_to_dir[set_key]
        rel = total_rel_band(file_handle, year, hist_name, set_dir, nuisances, mc_nom_y)

        lbl = _norm_unc_label(set_key)
        y_low  = mc_nom_y * (1.0 - rel)
        y_high = mc_nom_y * (1.0 + rel)
        bands.append((lbl, y_low, y_high, rel))

    # --- Figure with two pads (top: spectra, bottom: ratio) ---
    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    nano_color = "C0"
    nom_color  = "C1"

    # --- TOP PAD: Data/MC + filled uncertainty envelopes ---
    if data_y_nano is not None:
        ax.errorbar(x, data_y_nano, fmt="o", ms=3.5, capsize=0,
                    label="Data (NanoAOD)", color=nano_color)
    ax.errorbar(x, data_y, fmt="o", ms=3.5, capsize=0,
                label="Data (JES Nominal)", color=nom_color)

    if mc_nom_nano_y is not None:
        ax.step(e_nano[:-1], mc_nom_nano_y, where="post", linewidth=1.8,
                label="MC (NanoAOD)", color=nano_color)
    ax.step(e[:-1], mc_nom_y, where="post", linewidth=1.8,
            label="MC (JES+JER Nominal)", color=nom_color)

    order = {name: i for i, name in enumerate(
        ["JES Reduced", "JES Total", "JES Full", "JER Total", "JER Full"]
    )}
    bands.sort(key=lambda t: order.get(t[0], 99))

    for lbl, lo, hi, rel in bands:
        color = _color_for(lbl)
        ax.fill_between(
            x, lo, hi, step="mid",
            facecolor=color, edgecolor=color, alpha=0.45, linewidth=0.8,
            label="MC Unc: " + lbl
        )


    ax.set_title(f"{hist_name} — {year} / {era} (shape-normalized)")
    ax.set_ylabel("Normalized entries")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9, frameon=True)

    # --- BOTTOM PAD: ratio of envelopes to nominal (1 ± rel) ---
    # BOTTOM PAD header (keep the rest the same)
    rax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", label="0.0")

    # choose a sensible y-range from max rel across sets
    max_rel = 0.0
    for _, _, _, rel in bands:
        if rel is not None and np.size(rel):
            max_rel = max(max_rel, float(np.nanmax(rel)))
    # pad a bit; cap extremes to keep view reasonable
    pad = 0.02
    max_rel = min(max_rel, 0.5)  # do not explode axis if something goes wild
    #rax.set_ylim(1.0 - (max_rel + pad), 1.0 + (max_rel + pad))
    rax.set_ylim(-15, +15)

    # draw 1±rel lines for each set (only one legend entry per set — on top pad)
    for lbl, _, _, rel in bands:
        color = _color_for(lbl)
        hi =100*(0.0 + rel)
        lo =100*(0.0 - rel)
        # label only the solid (1+rel) line so we get one legend entry per set
        rax.step(e[:-1], hi, where="post", linewidth=1.3, color=color, label=f"{lbl}")
        rax.step(e[:-1], lo, where="post", linewidth=1.3, color=color, linestyle="--", label="_nolegend_")
    rax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    rax.set_xlabel("pT [GeV]" if "JetPt" in hist_name else hist_name)
    rax.set_ylabel("Uncertainty (%)")
    rax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print(f"Added page for {hist_name} {year} {era}")


def main():
    ap = argparse.ArgumentParser(description="Overlay Data vs MC with JES/JER bands (shape-only).")
    ap.add_argument("--config", default="JercFileAndTagNamesAK4.json", help="Path to JSON config (your JERC map).")
    ap.add_argument("--root",   default="output.root", help="Path to output.root produced by your job.")
    ap.add_argument("--hist",   default="hJetPt_AK4", choices=["hJetPt_AK4","hJetPt_AK8","hMET"],
                    help="Histogram name to plot (must exist in the ROOT file).")
    ap.add_argument("--sets",   nargs="+",
                    default=["ForUncertaintyJESReduced","ForUncertaintyJESTotal","ForUncertaintyJESFull",
                             "ForUncertaintyJERTotal","ForUncertaintyJERFull"],
                    help="Which sets to overlay (any subset of the defaults).")
    ap.add_argument("--years",  nargs="*", default=None,
                    help="Limit to these years; default = all years in config.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    years = args.years if args.years else list(cfg.keys())
    missing = []

    # Output PDF in current directory
    out_pdf = f"{os.path.splitext(os.path.basename(args.root))[0]}_{args.hist}.pdf"

    with uproot.open(args.root) as rf, PdfPages(out_pdf) as pdf:
        for year in years:
            if year not in cfg:
                print(f"Config has no year '{year}', skipping.")
                continue

            cfg_year = cfg[year]
            eras = list(cfg_year.get("data", {}).keys())
            if not eras:
                print(f"[{year}] No data eras found in config; skipping year.")
                continue

            for era in eras:
                mc_nom_path = f"{year}/MC/Nominal/Nominal/{args.hist}"
                data_nom_path = f"{year}/Data/{era}/Nominal/Nominal/{args.hist}"

                if mc_nom_path not in rf:
                    missing.append(mc_nom_path)
                    continue
                if data_nom_path not in rf:
                    missing.append(data_nom_path)
                    continue

                plot_one_year(rf, cfg_year, year, era, args.hist, pdf, args.sets)

    if missing:
        print("\nMissing objects/paths:")
        for m in missing:
            print("  -", m)

    print(f"Saved multipage PDF: {out_pdf}")

if __name__ == "__main__":
    main()

