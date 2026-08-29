import json
from pathlib import Path

import numpy as np


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


def find_metrics_path() -> Path:
    kaggle_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    for root in kaggle_roots:
        if root.exists():
            matches = list(root.rglob("mnist_all_stages_metrics.json"))
            if matches:
                return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
    raise FileNotFoundError(
        "No se encontro mnist_all_stages_metrics.json bajo /kaggle/working ni "
        "/kaggle/input."
    )


METRICS_PATH = find_metrics_path()
COLUMN_HEADER = f"{'Precision base':<18}{'E[R]':<16}{'sigma[R]':<16}{'Mediana':<16}{'Moda':<16}{'H':<16}"
STAT_LABELS_MD = ["Precisión base", "E[R]", "σ[R]", "Mediana", "Moda", "H"]


def fmt(agg: dict[str, tuple[float, float]], key: str, decimals: int = 4) -> str:
    m, s = agg[key]
    return f"{m:.{decimals}f}\u00b1{s:.{decimals}f}"


def fmt_pct(m: float, s: float, decimals: int = 2) -> str:
    return f"{100*m:.{decimals}f}%\u00b1{100*s:.{decimals}f}pp"


def fmt_row(baseline_m: float, baseline_s: float, agg: dict[str, tuple[float, float]]) -> str:
    return (f"{fmt_pct(baseline_m, baseline_s):<18}{fmt(agg,'mean'):<16}{fmt(agg,'std'):<16}"
            f"{fmt(agg,'median'):<16}{fmt(agg,'mode'):<16}{fmt(agg,'H'):<16}")


def tabla_descongelado_progresivo() -> list[dict]:
    data = load_json(METRICS_PATH)
    architecture = data["config"]["architecture"]
    dataset = data["config"]["datasets"][0]

    print(f"\n=== Tabla de SRS por etapa de descongelado (sumatorio discreto) -- {architecture} -- {dataset} ===")
    print(f"{'Stage':<7}{'% backbone':<12}{COLUMN_HEADER}")

    rows_out = []
    md_rows = []
    for stage_result in data["results"]:
        stage = stage_result["stage"]
        blocks = ", ".join(stage_result["unfrozen_blocks"]) if stage_result["unfrozen_blocks"] else "(solo cabeza)"
        frac = stage_result["fraccion_backbone_descongelado"]

        per_seed_stats = [discrete_stats(*srf_from_rows(rows)) for rows in stage_result["per_seed_rows"]]
        agg = aggregate(per_seed_stats)
        baseline_m = stage_result["aggregate_stats"]["baseline_accuracy"]["media"]
        baseline_s = stage_result["aggregate_stats"]["baseline_accuracy"]["std"]

        print(f"{stage:<7}{frac*100:<12.1f}{fmt_row(baseline_m, baseline_s, agg)}")

        rows_out.append({
            "stage": stage, "bloques_descongelados": blocks, "fraccion_backbone_descongelado": frac,
            "baseline_accuracy": {"media": baseline_m, "std": baseline_s},
            "estadisticos": {k: {"media": v[0], "std": v[1]} for k, v in agg.items()},
        })
        md_rows.append(
            f"| {stage} | {blocks} | {frac*100:.1f}% | {fmt_pct(baseline_m, baseline_s)} "
            f"| {fmt(agg,'mean')} | {fmt(agg,'std')} | {fmt(agg,'median')} | {fmt(agg,'mode')} | {fmt(agg,'H')} |"
        )

    md = [
        f"# Tabla de SRS por etapa de descongelado (sumatorio discreto) -- {architecture} -- {dataset}",
        "",
        "| Stage | Bloques descongelados | % backbone descongelado | Precisión base | " + " | ".join(STAT_LABELS_MD[1:]) + " |",
        "|---|---|---|---|---|---|---|---|---|",
    ] + md_rows

    output_dir = Path("/kaggle/working")
    (output_dir / f"{dataset}_srs_table_sumatorio.md").write_text("\n".join(md), encoding="utf-8")
    with (output_dir / f"{dataset}_srs_table_sumatorio.json").open("w", encoding="utf-8") as fh:
        json.dump(rows_out, fh, indent=2, ensure_ascii=False)

    print(f"\nGuardado en: {output_dir.resolve()}")
    print(f"  - {dataset}_srs_table_sumatorio.md")
    print(f"  - {dataset}_srs_table_sumatorio.json")
    return rows_out


if __name__ == "__main__":
    tabla_descongelado_progresivo()
