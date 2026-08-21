import json
from pathlib import Path

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


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
    p = np.diff(srf)
    r_i = x[1:]

    mean = float(np.sum(r_i * p))
    variance = float(np.sum((r_i - mean) ** 2 * p))
    std = float(np.sqrt(variance))

    median_mask = srf[1:] >= 0.5
    median = float(r_i[np.argmax(median_mask)]) if median_mask.any() else float(r_i[-1])

    mode_idx = int(np.argmax(p))
    mode = float(r_i[mode_idx])

    log_p = np.where(p > 0, np.log(p), 0.0)
    H = float(-np.sum(p * log_p))

    return {"mean": mean, "std": std, "median": median, "mode": mode, "H": H}


def aggregate(per_seed_stats: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    keys = per_seed_stats[0].keys()
    return {k: (float(np.mean([s[k] for s in per_seed_stats])), float(np.std([s[k] for s in per_seed_stats])))
            for k in keys}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def stats_from_per_seed_rows(per_seed_rows: list[list[dict]]) -> dict[str, tuple[float, float]]:
    per_seed_stats = []
    for rows in per_seed_rows:
        stats = discrete_stats(*srf_from_rows(rows))
        stats["baseline_accuracy"] = float(rows[-1]["accuracy"])
        per_seed_stats.append(stats)
    return aggregate(per_seed_stats)


def stats_from_per_seed(per_seed: list[dict]) -> dict[str, tuple[float, float]]:
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


COLUMN_HEADER = f"{'Precision base':<18}{'E[R]':<16}{'sigma[R]':<16}{'Mediana':<16}{'Moda':<16}{'H':<16}"


def fmt_row(agg: dict[str, tuple[float, float]]) -> str:
    return (f"{fmt_pct(agg):<18}{fmt(agg,'mean'):<16}{fmt(agg,'std'):<16}{fmt(agg,'median'):<16}"
            f"{fmt(agg,'mode'):<16}{fmt(agg,'H'):<16}")


LOCAL_PATHS = {
    "cnn_srf": Path(r"C:\Users\stst1\Downloads\outputs_TFG\outputs_srf_cnn_pequena\20260728_171111"),
    "resnet_srf": Path(r"C:\Users\stst1\Downloads\outputs_TFG\outputs_srf_resnet18_preentrenada\20260728_171133"),
    "stability": Path(r"C:\Users\stst1\Downloads\outputs_TFG\outputs_srf_estabilidad_semillas\20260728_140957"),
    "activation": Path(r"C:\Users\stst1\Downloads\outputs_TFG\outputs_srf_efecto_activacion\20260728_140848"),
    "autoencoder": Path(r"C:\Users\stst1\Downloads\outputs_TFG\outputs_srf_autoencoder\20260728_140809"),
    "vit": Path(r"C:\Users\stst1\Downloads\outputs_TFG\outputs_srf_vit\20260728_170918"),
}

FOLDER_NAMES = {
    "cnn_srf": ["spectral_cdf_outputs_en", "spectral_cdf_outputs"],
    "resnet_srf": ["resnet18_spectral_cdf_outputs_en", "resnet18_spectral_cdf_outputs"],
    "stability": ["srf_stability_outputs_en", "srf_stability_outputs"],
    "activation": ["srf_activation_outputs_en", "srf_activation_outputs"],
    "autoencoder": ["srf_autoencoder_outputs_en", "srf_autoencoder_outputs"],
    "vit": ["srf_vit_outputs_en", "srf_vit_outputs"],
}


def find_output_dir(roots: list[Path], folder_names: list[str]) -> Path | None:
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
AUTOENCODER_DIR = _PATHS["autoencoder"]
VIT_DIR = _PATHS["vit"]

print("Carpetas detectadas:")
for _name, _path in _PATHS.items():
    print(f"  {_name}: {_path}")


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/estadisticos_discretos_srf")
    return Path(r"C:\Users\stst1\Downloads\estadisticos_discretos_srf")


OUTPUT_DIR = default_output_dir()

DATASETS = ["mnist", "fashion_mnist", "kmnist", "cifar10"]
DISPLAY = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST", "cifar10": "CIFAR-10 (gris)"}

STAT_KEYS = ["baseline_accuracy", "mean", "std", "median", "mode", "H"]
STAT_LABELS_MD = ["Precisión base", "E[R]", "σ[R]", "Mediana", "Moda", "H"]


def agg_to_json(agg: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {k: {"media": v[0], "std": v[1]} for k, v in agg.items()}


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


def tabla_3_1() -> list[dict]:
    print("\n=== Tabla 3.1: Estadisticos de la SRF por conjunto de datos y arquitectura ===")
    print(f"{'Arquitectura':<12}{'Dataset':<16}{COLUMN_HEADER}")
    rows = []
    for net_name, base_dir in [("CNN", CNN_SRF_DIR), ("ResNet-18", RESNET_SRF_DIR)]:
        for ds in DATASETS:
            d = load_json(base_dir / f"{ds}_metrics.json")
            agg = stats_from_per_seed_rows(d["per_seed_rows"])
            print(f"{net_name:<12}{DISPLAY[ds]:<16}{fmt_row(agg)}")
            rows.append({"arquitectura": net_name, "dataset": DISPLAY[ds], "estadisticos": agg_to_json(agg)})
    return rows


def tabla_4_1() -> list[dict]:
    print("\n=== Tabla 4.1: Estabilidad de los estadisticos de la SRF frente a la semilla (MNIST, 10 semillas) ===")
    print(f"{'Arquitectura':<12}{COLUMN_HEADER}")
    rows = []
    for net_name, fname in [("CNN", "cnn_stability_metrics.json"), ("ResNet-18", "resnet18_stability_metrics.json")]:
        d = load_json(STABILITY_DIR / fname)
        agg = stats_from_per_seed(d["per_seed"])
        print(f"{net_name:<12}{fmt_row(agg)}")
        rows.append({"arquitectura": net_name, "estadisticos": agg_to_json(agg)})
    return rows


def tabla_4_2() -> list[dict]:
    print("\n=== Tabla 4.2: Estadisticos de la SRF segun la activacion de la CNN y referencia ResNet-18 (MNIST) ===")
    print(f"{'Modelo':<16}{COLUMN_HEADER}")
    rows = []
    for label, fname in [("ReLU", "relu_metrics.json"), ("GELU", "gelu_metrics.json"),
                          ("tanh", "tanh_metrics.json"), ("Sigmoide", "sigmoid_metrics.json"),
                          ("ResNet-18 ref.", "resnet18_reference_metrics.json")]:
        d = load_json(ACTIVATION_DIR / fname)
        agg = stats_from_per_seed(d["per_seed"])
        print(f"{label:<16}{fmt_row(agg)}")
        rows.append({"modelo": label, "estadisticos": agg_to_json(agg)})
    return rows


def tabla_5_2() -> list[dict]:
    print("\n=== Tabla 5.2: Estadisticos de la SRF del autoencoder por conjunto de datos ===")
    print(f"{'Dataset':<16}{'n semillas':<12}{COLUMN_HEADER}")
    rows = []
    for ds in DATASETS:
        d = load_json(AUTOENCODER_DIR / f"{ds}_metrics.json")
        n_seeds = len(d["per_seed"])
        agg = stats_from_per_seed(d["per_seed"])
        print(f"{DISPLAY[ds]:<16}{n_seeds:<12}{fmt_row(agg)}")
        rows.append({"dataset": DISPLAY[ds], "n_semillas": n_seeds, "estadisticos": agg_to_json(agg)})
    return rows


def tabla_5_3() -> list[dict]:
    print("\n=== Tabla 5.3: Estadisticos de la SRF del ViT-B/16 congelado por conjunto de datos ===")
    print(f"{'Dataset':<16}{'n semillas':<12}{COLUMN_HEADER}")
    rows = []
    for ds in DATASETS:
        d = load_json(VIT_DIR / f"{ds}_metrics.json")
        n_seeds = len(d["per_seed"])
        agg = stats_from_per_seed(d["per_seed"])
        print(f"{DISPLAY[ds]:<16}{n_seeds:<12}{fmt_row(agg)}")
        rows.append({"dataset": DISPLAY[ds], "n_semillas": n_seeds, "estadisticos": agg_to_json(agg)})
    return rows


def guardar_resultados(resultados: dict[str, list[dict]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "estadisticos_discretos_srf.json").open("w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)

    md = ["# Estadisticos discretos de la SRF (Capitulos 3-5)", ""]

    md += ["## Tabla 3.1 — SRF por conjunto de datos y arquitectura", ""]
    md.append(markdown_table(
        ["Arquitectura", "Dataset"] + STAT_LABELS_MD,
        [[r["arquitectura"], r["dataset"]] + agg_to_md_cells({k: (v["media"], v["std"]) for k, v in r["estadisticos"].items()})
         for r in resultados["tabla_3_1"]],
    ))

    md += ["", "## Tabla 4.1 — Estabilidad frente a la semilla (MNIST, 10 semillas)", ""]
    md.append(markdown_table(
        ["Arquitectura"] + STAT_LABELS_MD,
        [[r["arquitectura"]] + agg_to_md_cells({k: (v["media"], v["std"]) for k, v in r["estadisticos"].items()})
         for r in resultados["tabla_4_1"]],
    ))

    md += ["", "## Tabla 4.2 — Efecto de la activación (MNIST, 3 semillas)", ""]
    md.append(markdown_table(
        ["Modelo"] + STAT_LABELS_MD,
        [[r["modelo"]] + agg_to_md_cells({k: (v["media"], v["std"]) for k, v in r["estadisticos"].items()})
         for r in resultados["tabla_4_2"]],
    ))

    md += ["", "## Tabla 5.2 — Autoencoder por conjunto de datos", ""]
    md.append(markdown_table(
        ["Dataset", "n semillas"] + STAT_LABELS_MD,
        [[r["dataset"], str(r["n_semillas"])] + agg_to_md_cells({k: (v["media"], v["std"]) for k, v in r["estadisticos"].items()})
         for r in resultados["tabla_5_2"]],
    ))

    md += ["", "## Tabla 5.3 — ViT-B/16 congelado por conjunto de datos", ""]
    md.append(markdown_table(
        ["Dataset", "n semillas"] + STAT_LABELS_MD,
        [[r["dataset"], str(r["n_semillas"])] + agg_to_md_cells({k: (v["media"], v["std"]) for k, v in r["estadisticos"].items()})
         for r in resultados["tabla_5_3"]],
    ))

    (OUTPUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nGuardado en: {OUTPUT_DIR.resolve()}")
    print("  - report.md")
    print("  - estadisticos_discretos_srf.json")


if __name__ == "__main__":
    resultados = {
        "tabla_3_1": tabla_3_1(),
        "tabla_4_1": tabla_4_1(),
        "tabla_4_2": tabla_4_2(),
        "tabla_5_2": tabla_5_2(),
        "tabla_5_3": tabla_5_3(),
    }
    guardar_resultados(resultados)
