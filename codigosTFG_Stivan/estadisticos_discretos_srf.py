"""
Recomputes, discretely and consistently, every SRF statistic that appears in
the thesis tables of Chapters 3 and 4 (the ones with an English-language
Kaggle notebook):

    Table 3.1 (Ch. 3, CNN vs ResNet-18, 4 datasets, 3 seeds)
    Table 4.1 (Ch. 4, stability across seeds, MNIST, 10 seeds)
    Table 4.2 (Ch. 4, effect of the activation function, MNIST, 3 seeds)

Chapter 5's Tables 5.2 (autoencoder) and 5.3 (ViT-B/16) are NOT covered by
this script -- see the note at the bottom of this docstring.

Trains nothing and fabricates no numbers: it reads directly the *_metrics.json
files already downloaded from the Kaggle notebooks (the raw per-seed accuracy
at every point of the sweep, in "rows"/"per_seed_rows", with
"normalized_radius" and "accuracy").

Fixes two problems found in the original scripts:
  1. The statistics are computed on SRF(r) = (F(r)-F(0))/(1-F(0))
     (SRF(0)=0 by construction), not on the raw F(r).
  2. The entropy H is the discrete Shannon entropy over the point masses
     p_i = SRF(r_i) - SRF(r_{i-1}) (Appendix B of the thesis, formula 15 of
     the article), not the differential entropy (which can come out negative
     and is what fit_spectral_distributions*.py and the
     activation/autoencoder/ViT/stability scripts compute by mistake).

Methodology: same as the tables already published in the thesis -- the
statistics are computed per seed first and then aggregated as
mean +/- standard deviation across seeds (not the other way round, i.e. not
computed on the already-averaged curve).

This script can run either locally (reusing *_metrics.json files already
downloaded from the Kaggle results folders) or directly inside a Kaggle
notebook. On Kaggle, it auto-detects each experiment's output folder by name,
searching /kaggle/working (same-session run) and then /kaggle/input (results
attached as a data source: Add Data -> Your Work). If none is found, it falls
back to the local Windows paths below.

Note: the autoencoder and ViT experiments have no English-language Kaggle
script yet (only the Spanish originals, experimento_srf_autoencoder.py and
experimento_srf_vit.py), so their output folders are still named
"srf_autoencoder_outputs"/"srf_vit_outputs" (no "_en" suffix) even when read
from this English script.
"""
import json
from pathlib import Path

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def pool_adjacent_violators(y: np.ndarray) -> np.ndarray:
    values, weights = [], []
    for yi in y:
        values.append(float(yi))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            mw = weights[-2] + weights[-1]
            mv = (values[-2] * weights[-2] + values[-1] * weights[-1]) / mw
            values.pop(); weights.pop()
            values[-1] = mv; weights[-1] = mw
    fitted = np.empty(len(y), dtype=float)
    pos = 0
    for value, weight in zip(values, weights):
        n = int(round(weight))
        fitted[pos:pos + n] = value
        pos += n
    return fitted


def srf_from_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """From the raw sweep rows (normalized_radius, accuracy) of ONE seed,
    returns (x, SRF) already isotonic-corrected and renormalized at the
    origin (SRF(0)=0, SRF(1)=1)."""
    x = np.array([r["normalized_radius"] for r in rows], dtype=float)
    acc = np.array([r["accuracy"] for r in rows], dtype=float)
    acc_min, acc_max = acc[0], acc[-1]
    srf = np.clip((acc - acc_min) / max(acc_max - acc_min, 1e-12), 0.0, 1.0)
    srf = pool_adjacent_violators(srf)
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        srf = np.concatenate([[0.0], srf])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        srf = np.concatenate([srf, [1.0]])
    return x, srf


def discrete_stats(x: np.ndarray, srf: np.ndarray) -> dict[str, float]:
    """Discrete statistics of ONE already-renormalized SRF curve (one seed):
    E[R], sigma[R], median, mode, AUC=1-E[R], and the discrete Shannon
    entropy H over the point masses p_i = SRF_i - SRF_{i-1} (always >= 0)."""
    survival = 1.0 - srf
    mean = float(_trapz(survival, x))
    second_moment = float(2.0 * _trapz(x * survival, x))
    variance = max(second_moment - mean ** 2, 0.0)
    std = float(np.sqrt(variance))
    median = float(np.interp(0.5, srf, x))

    widths = np.diff(x)
    nz = widths > 1e-12
    density = np.zeros_like(widths)
    density[nz] = np.diff(srf)[nz] / widths[nz]
    mode_idx = int(np.argmax(density))
    mode = float(0.5 * (x[mode_idx] + x[mode_idx + 1]))

    auc = float(_trapz(srf, x))

    p = np.diff(srf, prepend=0.0)
    p = np.clip(p, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = float(-np.sum(p * log_p))

    return {"mean": mean, "std": std, "median": median, "mode": mode, "auc": auc, "H": H}


def aggregate(per_seed_stats: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Mean +/- standard deviation across seeds, for each statistic."""
    keys = per_seed_stats[0].keys()
    return {k: (float(np.mean([s[k] for s in per_seed_stats])), float(np.std([s[k] for s in per_seed_stats])))
            for k in keys}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def stats_from_per_seed_rows(per_seed_rows: list[list[dict]]) -> dict[str, tuple[float, float]]:
    """For the main SRF sweep schema (Ch. 3): 'per_seed_rows' is directly a
    list of lists of rows, one list per seed. There is no explicit
    'baseline_accuracy' field; the baseline accuracy is the last point of the
    sweep (full spectrum, r=1), which by construction equals Acc(f_M)."""
    per_seed_stats = []
    for rows in per_seed_rows:
        stats = discrete_stats(*srf_from_rows(rows))
        stats["baseline_accuracy"] = float(rows[-1]["accuracy"])
        per_seed_stats.append(stats)
    return aggregate(per_seed_stats)


def stats_from_per_seed(per_seed: list[dict]) -> dict[str, tuple[float, float]]:
    """For the stability/activation/autoencoder/ViT schema (Ch. 4 and 5):
    'per_seed' is a list of {seed, ..., rows, stats}. Uses the explicit
    'baseline_accuracy' field if present; otherwise falls back to the last
    point of the sweep, same as stats_from_per_seed_rows."""
    per_seed_stats = []
    for entry in per_seed:
        stats = discrete_stats(*srf_from_rows(entry["rows"]))
        stats["baseline_accuracy"] = float(entry.get("baseline_accuracy", entry["rows"][-1]["accuracy"]))
        per_seed_stats.append(stats)
    return aggregate(per_seed_stats)


def fmt(agg: dict[str, tuple[float, float]], key: str, decimals: int = 4) -> str:
    m, s = agg[key]
    return f"{m:.{decimals}f}\u00b1{s:.{decimals}f}"


def fmt_pct(agg: dict[str, tuple[float, float]], key: str = "baseline_accuracy", decimals: int = 2) -> str:
    m, s = agg[key]
    return f"{100*m:.{decimals}f}%\u00b1{100*s:.{decimals}f}pp"


# Identical columns for all 5 tables: Baseline accuracy, E[R], sigma[R],
# Median, Mode, AUC, H. If another statistic is needed in the future, add it
# here once and it propagates to all 5 tables automatically.
COLUMN_HEADER = f"{'Baseline acc.':<18}{'E[R]':<16}{'sigma[R]':<16}{'Median':<16}{'Mode':<16}{'AUC':<16}{'H':<16}"


def fmt_row(agg: dict[str, tuple[float, float]]) -> str:
    return (f"{fmt_pct(agg):<18}{fmt(agg,'mean'):<16}{fmt(agg,'std'):<16}{fmt(agg,'median'):<16}"
            f"{fmt(agg,'mode'):<16}{fmt(agg,'auc'):<16}{fmt(agg,'H'):<16}")


# ---------------------------------------------------------------------------
# Paths to the outputs already downloaded from Kaggle (local fallback only;
# see module docstring for the Kaggle auto-detection logic below).
# ---------------------------------------------------------------------------

LOCAL_PATHS = {
    "cnn_srf": Path(r"C:\Users\stst1\Downloads\results (52)\spectral_cdf_outputs\20260728_171111"),
    "resnet_srf": Path(r"C:\Users\stst1\Downloads\results (53)\resnet18_spectral_cdf_outputs\20260728_171133"),
    "stability": Path(r"C:\Users\stst1\Downloads\results (48)\srf_stability_outputs\20260728_140957"),
    "activation": Path(r"C:\Users\stst1\Downloads\results (47)\srf_activation_outputs\20260728_140848"),
}

# Folder names each experiment produces (English sweep scripts append "_en").
# Note: the autoencoder and ViT experiments (Chapter 5) have no English
# Kaggle script yet -- only experimento_srf_autoencoder.py and
# experimento_srf_vit.py (Spanish) exist -- so Tables 5.2/5.3 are
# intentionally NOT covered by this English script. Use the Spanish
# estadisticos_discretos_completos.py for those two, or
# english_estadisticos_discretos_completos.py once an English version of
# those two experiments exists.
FOLDER_NAMES = {
    "cnn_srf": ["spectral_cdf_outputs_en", "spectral_cdf_outputs"],
    "resnet_srf": ["resnet18_spectral_cdf_outputs_en", "resnet18_spectral_cdf_outputs"],
    "stability": ["srf_stability_outputs_en", "srf_stability_outputs"],
    "activation": ["srf_activation_outputs_en", "srf_activation_outputs"],
}


def find_output_dir(roots: list[Path], folder_names: list[str]) -> Path | None:
    """Search `roots` (in order) for a directory literally named one of
    `folder_names`, and return its most recent timestamped run subfolder
    (the one holding the *_metrics.json files). Kaggle mounts an attached
    notebook's output under an unpredictable slug path, so this searches by
    name instead of requiring the exact path. Returns None if nothing is
    found."""
    for folder_name in folder_names:
        for root in roots:
            if not root.exists():
                continue
            matches = [p for p in root.rglob(folder_name) if p.is_dir()]
            if matches:
                base = matches[0]
                subdirs = [d for d in base.iterdir() if d.is_dir()]
                return sorted(subdirs, key=lambda d: d.name)[-1] if subdirs else base
    return None


def resolve_paths() -> dict[str, Path]:
    kaggle_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    on_kaggle = any(root.exists() for root in kaggle_roots)
    resolved = {}
    for key, folder_names in FOLDER_NAMES.items():
        found = find_output_dir(kaggle_roots, folder_names) if on_kaggle else None
        resolved[key] = found if found is not None else LOCAL_PATHS[key]
    return resolved


_PATHS = resolve_paths()
CNN_SRF_DIR = _PATHS["cnn_srf"]
RESNET_SRF_DIR = _PATHS["resnet_srf"]
STABILITY_DIR = _PATHS["stability"]
ACTIVATION_DIR = _PATHS["activation"]

print("Detected folders:")
for _name, _path in _PATHS.items():
    print(f"  {_name}: {_path}")


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/discrete_srf_statistics")
    return Path(r"C:\Users\stst1\Downloads\discrete_srf_statistics")


OUTPUT_DIR = default_output_dir()

DATASETS = ["mnist", "fashion_mnist", "kmnist", "cifar10"]
DISPLAY = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST", "cifar10": "CIFAR-10 (grayscale)"}


# ---------------------------------------------------------------------------
# Export helpers: markdown (report.md) and JSON (raw numbers, not just
# formatted strings, so they can be reused without re-parsing text).
# ---------------------------------------------------------------------------

STAT_KEYS = ["baseline_accuracy", "mean", "std", "median", "mode", "auc", "H"]
STAT_LABELS_MD = ["Baseline accuracy", "E[R]", "σ[R]", "Median", "Mode", "AUC", "H"]


def agg_to_json(agg: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()}


def agg_to_md_cells(agg: dict[str, tuple[float, float]]) -> list[str]:
    cells = [fmt_pct(agg)]
    for key in STAT_KEYS[1:]:
        cells.append(fmt(agg, key))
    return cells


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3.1 - CNN vs ResNet-18, 4 datasets
# ---------------------------------------------------------------------------

def table_3_1() -> list[dict]:
    print("\n=== Table 3.1: SRF statistics by dataset and architecture ===")
    print(f"{'Architecture':<14}{'Dataset':<16}{COLUMN_HEADER}")
    rows = []
    for net_name, base_dir in [("CNN", CNN_SRF_DIR), ("ResNet-18", RESNET_SRF_DIR)]:
        for ds in DATASETS:
            d = load_json(base_dir / f"{ds}_metrics.json")
            agg = stats_from_per_seed_rows(d["per_seed_rows"])
            print(f"{net_name:<14}{DISPLAY[ds]:<16}{fmt_row(agg)}")
            rows.append({"architecture": net_name, "dataset": DISPLAY[ds], "statistics": agg_to_json(agg)})
    return rows


# ---------------------------------------------------------------------------
# Table 4.1 - Stability across seeds, MNIST, 10 seeds
# ---------------------------------------------------------------------------

def table_4_1() -> list[dict]:
    print("\n=== Table 4.1: Stability of the SRF statistics across seeds (MNIST, 10 seeds) ===")
    print(f"{'Architecture':<14}{COLUMN_HEADER}")
    rows = []
    for net_name, fname in [("CNN", "cnn_stability_metrics.json"), ("ResNet-18", "resnet18_stability_metrics.json")]:
        d = load_json(STABILITY_DIR / fname)
        agg = stats_from_per_seed(d["per_seed"])
        print(f"{net_name:<14}{fmt_row(agg)}")
        rows.append({"architecture": net_name, "statistics": agg_to_json(agg)})
    return rows


# ---------------------------------------------------------------------------
# Table 4.2 - Effect of the activation function, MNIST, 3 seeds + ResNet-18 reference
# ---------------------------------------------------------------------------

def table_4_2() -> list[dict]:
    print("\n=== Table 4.2: SRF statistics by CNN activation, with ResNet-18 reference (MNIST) ===")
    print(f"{'Model':<16}{COLUMN_HEADER}")
    rows = []
    for label, fname in [("ReLU", "relu_metrics.json"), ("GELU", "gelu_metrics.json"),
                          ("tanh", "tanh_metrics.json"), ("Sigmoid", "sigmoid_metrics.json"),
                          ("ResNet-18 ref.", "resnet18_reference_metrics.json")]:
        d = load_json(ACTIVATION_DIR / fname)
        agg = stats_from_per_seed(d["per_seed"])
        print(f"{label:<16}{fmt_row(agg)}")
        rows.append({"model": label, "statistics": agg_to_json(agg)})
    return rows


# Note: Tables 5.2 (autoencoder) and 5.3 (ViT-B/16) are intentionally not
# covered here -- see the FOLDER_NAMES comment above.


def save_results(results: dict[str, list[dict]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "discrete_srf_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    md = ["# Discrete SRF statistics (Chapters 3-4)", ""]

    md += ["## Table 3.1 — SRF by dataset and architecture", ""]
    md.append(markdown_table(
        ["Architecture", "Dataset"] + STAT_LABELS_MD,
        [[r["architecture"], r["dataset"]] + agg_to_md_cells({k: (v["mean"], v["std"]) for k, v in r["statistics"].items()})
         for r in results["table_3_1"]],
    ))

    md += ["", "## Table 4.1 — Stability across seeds (MNIST, 10 seeds)", ""]
    md.append(markdown_table(
        ["Architecture"] + STAT_LABELS_MD,
        [[r["architecture"]] + agg_to_md_cells({k: (v["mean"], v["std"]) for k, v in r["statistics"].items()})
         for r in results["table_4_1"]],
    ))

    md += ["", "## Table 4.2 — Effect of the activation function (MNIST, 3 seeds)", ""]
    md.append(markdown_table(
        ["Model"] + STAT_LABELS_MD,
        [[r["model"]] + agg_to_md_cells({k: (v["mean"], v["std"]) for k, v in r["statistics"].items()})
         for r in results["table_4_2"]],
    ))

    md += ["", "---", "",
           "Note: Tables 5.2 (autoencoder) and 5.3 (ViT-B/16) are not included "
           "in this English script, because those two experiments have no "
           "English-language Kaggle notebook yet (only the Spanish originals, "
           "experimento_srf_autoencoder.py and experimento_srf_vit.py). Use "
           "estadisticos_discretos_completos.py (Spanish) for those two tables."]

    (OUTPUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")
    print("  - report.md (Tables 3.1, 4.1, 4.2 in markdown, to read directly)")
    print("  - discrete_srf_statistics.json (the same numbers, raw)")


if __name__ == "__main__":
    results = {
        "table_3_1": table_3_1(),
        "table_4_1": table_4_1(),
        "table_4_2": table_4_2(),
    }
    save_results(results)
