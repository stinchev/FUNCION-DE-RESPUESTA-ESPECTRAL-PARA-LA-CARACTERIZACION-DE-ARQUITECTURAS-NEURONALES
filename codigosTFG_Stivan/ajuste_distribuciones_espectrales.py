"""
Ajusta familias de distribuciones parametricas (Normal, Exponencial, Logistica,
Beta) a las CDF espectrales empiricas ya calculadas por los experimentos de
barrido espectral acumulativo (kaggle_vision_spectral_cdf_experiment.py para la
CNN pequena, kaggle_vision_resnet18_spectral_cdf_experiment.py para ResNet-18
congelada), para ver a que distribucion se parece cada red en cada dataset y
sacar sus parametros.

Normal/Exponencial/Logistica se truncan y renormalizan a soporte [0,1]
(F_trunc(x) = (F0(x)-F0(0)) / (F0(1)-F0(0))) para que sean CDFs validas en
nuestro dominio. Beta ya tiene soporte nativo en [0,1], no hace falta truncar.

El ajuste minimiza el error cuadratico entre la CDF candidata y la CDF empirica
(la media entre las 3 semillas del ajuste isotonico ya calculado en los scripts
de Kaggle), usando como punto de partida el metodo de los momentos con la
media/varianza empirica.

Este script puede ejecutarse tanto en local (reutilizando los *_metrics.json
ya descargados de las carpetas de resultados de Kaggle) como directamente
dentro de un notebook de Kaggle. En Kaggle, detecta automaticamente las
carpetas de salida de los dos experimentos de barrido ("spectral_cdf_outputs"
y "resnet18_spectral_cdf_outputs") buscando primero en /kaggle/working (si
se ejecutaron antes en esta misma sesion) y luego en /kaggle/input (si se
han adjuntado como fuente de datos: Add Data -> Your Work -> elegir la
version del notebook cuya pestana Output tenga los resultados). Si no
encuentra ninguna, recurre a las rutas locales de Windows en LOCAL_NETWORKS.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

_trapz = getattr(np, "trapezoid", None) or np.trapz

# Alternativa local (no-Kaggle): ajustar estas rutas a las carpetas de
# resultados descargadas de Kaggle, si se ejecuta este script en el propio
# ordenador.
LOCAL_NETWORKS = {
    "CNN pequena": Path(r"C:\Users\stst1\Downloads\results (52)\spectral_cdf_outputs\20260728_171111"),
    "ResNet-18 congelada": Path(r"C:\Users\stst1\Downloads\results (53)\resnet18_spectral_cdf_outputs\20260728_171133"),
}
LOCAL_OUTPUT_DIR = Path(r"C:\Users\stst1\Downloads\spectral_distribution_fits")

DATASETS = ["mnist", "fashion_mnist", "kmnist", "cifar10"]
DISPLAY = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST", "cifar10": "CIFAR-10 (gris)"}


def find_output_dir(roots: list[Path], folder_names: list[str]) -> Path:
    """Busca en `roots` (en orden) un directorio llamado exactamente uno de
    `folder_names` (los scripts en ingles anaden "_en" al nombre de la
    carpeta de salida, p. ej. "spectral_cdf_outputs_en", mientras que los
    originales en espanol no -- se prueban ambas variantes), y devuelve su
    subcarpeta con fecha mas reciente (la que realmente contiene los
    *_metrics.json). Kaggle monta la salida de un notebook adjunto bajo una
    ruta con un slug impredecible, asi que se busca por nombre en vez de
    exigir la ruta exacta."""
    for folder_name in folder_names:
        for root in roots:
            if not root.exists():
                continue
            matches = [p for p in root.rglob(folder_name) if p.is_dir()]
            if matches:
                base = matches[0]
                subdirs = [d for d in base.iterdir() if d.is_dir()]
                return sorted(subdirs, key=lambda d: d.name)[-1] if subdirs else base
    raise FileNotFoundError(
        f"No se ha encontrado ninguna carpeta llamada {folder_names} bajo {[str(r) for r in roots]}. "
        f"O bien ejecuta antes el script de barrido correspondiente en esta misma sesion de Kaggle, "
        f"o adjunta su salida como fuente de datos (Add Data -> Your Work -> elegir la version del "
        f"notebook con los resultados) y vuelve a ejecutar."
    )


def default_networks() -> dict[str, Path]:
    kaggle_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    if any(root.exists() for root in kaggle_roots):
        try:
            return {
                "CNN pequena": find_output_dir(kaggle_roots, ["spectral_cdf_outputs_en", "spectral_cdf_outputs"]),
                "ResNet-18 congelada": find_output_dir(
                    kaggle_roots, ["resnet18_spectral_cdf_outputs_en", "resnet18_spectral_cdf_outputs"]
                ),
            }
        except FileNotFoundError as exc:
            print(f"{exc}\nUsando LOCAL_NETWORKS como alternativa.")
    return LOCAL_NETWORKS


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/spectral_distribution_fits")
    return LOCAL_OUTPUT_DIR


NETWORKS = default_networks()
OUTPUT_DIR = default_output_dir()
print(f"Redes detectadas: {[(name, str(path)) for name, path in NETWORKS.items()]}")
print(f"Directorio de salida: {OUTPUT_DIR}")


def renormalize_srf(y: np.ndarray) -> np.ndarray:
    """SRF(r) = (F(r)-F(0)) / (1-F(0)), para que SRF(0)=0 exactamente."""
    y = np.asarray(y, dtype=float)
    y0 = y[0]
    denom = max(1.0 - y0, 1e-12)
    return np.clip((y - y0) / denom, 0.0, 1.0)


def load_empirical_cdf(base_dir: Path, dataset: str):
    """Carga la SRF(r) empirica (media de las semillas), ya renormalizada en
    el origen (SRF(0)=0), que es la que se ajusta y se representa en las
    figuras del TFG."""
    with (base_dir / f"{dataset}_metrics.json").open(encoding="utf-8") as fh:
        result = json.load(fh)
    rows0 = result["per_seed_rows"][0]
    x = np.array([r["normalized_radius"] for r in rows0], dtype=float)
    iso_curves = np.array(result["per_seed_isotonic_cdf"], dtype=float)
    srf_curves = np.array([renormalize_srf(c) for c in iso_curves])
    f_mean = srf_curves.mean(axis=0)
    # anadir los extremos (0,0) y (1,1) igual que en cdf_statistics, para que
    # el ajuste tambien respete el borde del soporte
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        f_mean = np.concatenate([[0.0], f_mean])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        f_mean = np.concatenate([f_mean, [1.0]])
    return x, f_mean


def truncated_cdf(base_cdf_fn, x, params):
    f0 = base_cdf_fn(x, *params)
    f0_0 = base_cdf_fn(np.array([0.0]), *params)[0]
    f0_1 = base_cdf_fn(np.array([1.0]), *params)[0]
    denom = max(f0_1 - f0_0, 1e-12)
    return np.clip((f0 - f0_0) / denom, 0.0, 1.0)


def beta_moment_init(mean, std):
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    var = min(std ** 2, mean * (1 - mean) * 0.98)
    common = mean * (1 - mean) / var - 1
    a = max(mean * common, 1e-2)
    b = max((1 - mean) * common, 1e-2)
    return [a, b]


FAMILIES = {
    "Normal": {
        "base_cdf": lambda x, mu, sigma: stats.norm.cdf(x, loc=mu, scale=sigma),
        "init": lambda mean, std: [mean, max(std, 1e-3)],
        "bounds": [(-2.0, 3.0), (1e-3, 5.0)],
        "param_names": ["mu", "sigma"],
        "truncate": True,
    },
    "Exponencial": {
        "base_cdf": lambda x, scale: stats.expon.cdf(x, loc=0.0, scale=scale),
        "init": lambda mean, std: [max(mean, 1e-3)],
        "bounds": [(1e-3, 5.0)],
        "param_names": ["scale (1/lambda)"],
        "truncate": True,
    },
    "Logistica": {
        "base_cdf": lambda x, mu, s: stats.logistic.cdf(x, loc=mu, scale=s),
        "init": lambda mean, std: [mean, max(std * np.sqrt(3) / np.pi, 1e-3)],
        "bounds": [(-2.0, 3.0), (1e-3, 5.0)],
        "param_names": ["mu", "s"],
        "truncate": True,
    },
    "Beta": {
        "base_cdf": lambda x, a, b: stats.beta.cdf(x, a, b),
        "init": lambda mean, std: beta_moment_init(mean, std),
        "bounds": [(1e-2, 200.0), (1e-2, 200.0)],
        "param_names": ["alpha", "beta"],
        "truncate": False,
    },
}


def cdf_extra_stats(x: np.ndarray, f: np.ndarray, mean: float, var: float) -> dict:
    """Descriptores adicionales de la SRF (Spectral Response Function), adaptando
    la terminologia del documento SFC.pdf: mediana, moda, entropia diferencial de
    la densidad implicita, area bajo la curva (AUC = 1 - E[r] por construccion) y
    el hueco de entropia frente a una Normal de la misma varianza (mide cuanto se
    aleja la forma real de una campana simple; una meseta o bimodalidad como la
    de ResNet-18 en MNIST se traduce en un hueco de entropia grande)."""
    median = float(np.interp(0.5, f, x))
    widths_raw = np.diff(x)
    nonzero = widths_raw > 1e-12  # descarta tramos de ancho 0 (empates de radio por simetria del espectro)
    density_raw = np.zeros_like(widths_raw)
    density_raw[nonzero] = np.diff(f)[nonzero] / widths_raw[nonzero]
    mode_idx = int(np.argmax(density_raw))
    mode = float(0.5 * (x[mode_idx] + x[mode_idx + 1]))

    widths = widths_raw[nonzero]
    density = density_raw[nonzero]
    total_mass = float(np.sum(density * widths))
    density_norm = density / total_mass if total_mass > 0 else density
    eps = 1e-12
    entropy = float(-np.sum(density_norm * widths * np.log(np.maximum(density_norm, eps))))

    auc = float(_trapz(f, x))  # = 1 - mean, se calcula igualmente por claridad/consistencia con el paper

    gaussian_entropy = float(0.5 * np.log(2 * np.pi * np.e * max(var, eps)))
    entropy_gap = gaussian_entropy - entropy

    return {
        "median": median, "mode": mode, "entropy": entropy,
        "auc": auc, "gaussian_entropy": gaussian_entropy, "entropy_gap": entropy_gap,
    }


def fit_family(name, spec, x, f_empirical, mean, std):
    init = spec["init"](mean, std)

    def candidate_cdf(params):
        if spec["truncate"]:
            return truncated_cdf(spec["base_cdf"], x, params)
        return spec["base_cdf"](x, *params)

    def loss(params):
        return float(np.sum((candidate_cdf(params) - f_empirical) ** 2))

    result = minimize(loss, x0=init, method="L-BFGS-B", bounds=spec["bounds"])
    params = result.x
    f_fit = candidate_cdf(params)
    sse = float(np.sum((f_fit - f_empirical) ** 2))
    ks = float(np.max(np.abs(f_fit - f_empirical)))
    return {
        "family": name,
        "params": dict(zip(spec["param_names"], [float(p) for p in params])),
        "sse": sse, "ks": ks, "fitted_curve": f_fit,
    }


def build_report(all_results: dict, output_dir: Path) -> None:
    lines = [
        "# Ajuste de distribuciones parametricas a la CDF espectral (CNN pequena vs ResNet-18 congelada)",
        "",
        "## Descriptores completos de la SRF (Spectral Response Function) por dataset y red",
        "",
        "| Red | Dataset | E[r] | std[r] | Mediana | Moda | AUC (=1-E[r]) | Entropía | Entropía Normal equiv. | Hueco de entropía |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (network_name, dataset), data in all_results.items():
        lines.append(
            f"| {network_name} | {DISPLAY[dataset]} | {data['mean']:.4f} | {data['std']:.4f} | "
            f"{data['median']:.4f} | {data['mode']:.4f} | {data['auc']:.4f} | {data['entropy']:.4f} | "
            f"{data['gaussian_entropy']:.4f} | {data['entropy_gap']:.4f} |"
        )
    lines += [
        "",
        "`E[r]` es el estadístico de referencia recomendado (un único número por dataset/red que resume "
        "\"cuánto espectro necesita\" esa combinación). El `hueco de entropía` (entropía de una Normal con "
        "la misma varianza menos la entropía real) es un diagnóstico de forma: valores altos indican que "
        "la SRF real es más estructurada/compleja (p.ej. meseta o bimodalidad) de lo que capturaría el "
        "par (E[r], std[r]) por sí solo — esto ocurre de forma notable en ResNet-18/MNIST.",
        "",
        "## Índice de desajuste espectral (E[r]_ResNet-18 / E[r]_CNN) por dataset",
        "",
        "| Dataset | E[r] CNN | E[r] ResNet-18 | Índice |",
        "|---|---|---|---|",
    ]
    cnn_name, resnet_name = list(NETWORKS.keys())
    for dataset in DATASETS:
        e_cnn = all_results[(cnn_name, dataset)]["mean"]
        e_resnet = all_results[(resnet_name, dataset)]["mean"]
        lines.append(f"| {DISPLAY[dataset]} | {e_cnn:.4f} | {e_resnet:.4f} | {e_resnet / e_cnn:.2f} |")

    lines += ["", "## Mejor familia por dataset y red", "", "| Red | Dataset | Mejor familia | Parámetros | KS | SSE |", "|---|---|---|---|---|---|"]
    for (network_name, dataset), data in all_results.items():
        best = data["fits"][0]
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in best["params"].items())
        lines.append(f"| {network_name} | {DISPLAY[dataset]} | {best['family']} | {params_str} | {best['ks']:.4f} | {best['sse']:.5f} |")

    lines += ["", "## SSE por familia (menor = mejor ajuste)", "", "| Red | Dataset | Normal | Exponencial | Logística | Beta |", "|---|---|---|---|---|---|"]
    for (network_name, dataset), data in all_results.items():
        sse_by_family = {f["family"]: f["sse"] for f in data["fits"]}
        lines.append(
            f"| {network_name} | {DISPLAY[dataset]} | {sse_by_family['Normal']:.5f} | "
            f"{sse_by_family['Exponencial']:.5f} | {sse_by_family['Logistica']:.5f} | {sse_by_family['Beta']:.5f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for network_name, base_dir in NETWORKS.items():
        for dataset in DATASETS:
            x, f_empirical = load_empirical_cdf(base_dir, dataset)
            mean = float(_trapz(1.0 - f_empirical, x))
            var = float(2.0 * _trapz(x * (1.0 - f_empirical), x) - mean ** 2)
            std = float(np.sqrt(max(var, 0.0)))
            extra = cdf_extra_stats(x, f_empirical, mean, var)

            fits = []
            for family_name, spec in FAMILIES.items():
                fits.append(fit_family(family_name, spec, x, f_empirical, mean, std))
            fits.sort(key=lambda r: r["sse"])
            all_results[(network_name, dataset)] = {
                "mean": mean, "std": std, **extra, "fits": fits, "x": x, "f_empirical": f_empirical,
            }

    print(f"{'Red':<22}{'Dataset':<16}{'E[r]':<8}{'std[r]':<8}{'entropia':<10}{'hueco H':<10}")
    print("-" * 64)
    for (network_name, dataset), data in all_results.items():
        print(f"{network_name:<22}{DISPLAY[dataset]:<16}{data['mean']:<8.4f}{data['std']:<8.4f}{data['entropy']:<10.4f}{data['entropy_gap']:<10.4f}")

    print("\nIndice de desajuste espectral (E[r] ResNet-18 / E[r] CNN):")
    cnn_name, resnet_name = list(NETWORKS.keys())
    for dataset in DATASETS:
        e_cnn = all_results[(cnn_name, dataset)]["mean"]
        e_resnet = all_results[(resnet_name, dataset)]["mean"]
        print(f"  {DISPLAY[dataset]:<16} indice={e_resnet/e_cnn:.2f}")

    print(f"\n{'Red':<22}{'Dataset':<16}{'Mejor familia':<14}{'Parametros':<38}{'KS':<8}{'SSE':<10}")
    print("-" * 108)
    for (network_name, dataset), data in all_results.items():
        best = data["fits"][0]
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in best["params"].items())
        print(f"{network_name:<22}{DISPLAY[dataset]:<16}{best['family']:<14}{params_str:<38}{best['ks']:<8.4f}{best['sse']:<10.5f}")

    print("\n\nDetalle de SSE por familia (menor = mejor ajuste):")
    print(f"{'Red':<22}{'Dataset':<16}{'Normal':<10}{'Exponencial':<13}{'Logistica':<12}{'Beta':<10}")
    print("-" * 83)
    for (network_name, dataset), data in all_results.items():
        sse_by_family = {f["family"]: f["sse"] for f in data["fits"]}
        print(f"{network_name:<22}{DISPLAY[dataset]:<16}"
              f"{sse_by_family['Normal']:<10.5f}{sse_by_family['Exponencial']:<13.5f}"
              f"{sse_by_family['Logistica']:<12.5f}{sse_by_family['Beta']:<10.5f}")

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, network_name in zip(axes, NETWORKS.keys()):
            data = all_results[(network_name, dataset)]
            ax.plot(data["x"], data["f_empirical"], "k.", ms=4, alpha=0.5, label="SRF empírica (media 3 semillas)")
            colors = {"Normal": "C0", "Exponencial": "C1", "Logistica": "C2", "Beta": "C3"}
            for fit in data["fits"]:
                ax.plot(data["x"], fit["fitted_curve"], color=colors[fit["family"]],
                        label=f"{fit['family']} (SSE={fit['sse']:.3f})", linewidth=1.6)
            ax.set_xlabel(f"r — {network_name}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("SRF(r)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"fit_comparison_{dataset}.png", dpi=150)
        plt.close()

    build_report(all_results, OUTPUT_DIR)
    print(f"\nGuardado en: {OUTPUT_DIR.resolve()}")
    return all_results


if __name__ == "__main__":
    main()
