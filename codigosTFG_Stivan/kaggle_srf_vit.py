"""
Anexo exploratorio: SRF de un Vision Transformer (ViT) preentrenado con el
backbone congelado (linear probe), en la misma línea que se hizo con la
ResNet-18 pero sin integrarlo en el análisis comparativo riguroso de los
Capítulos 2 y 3.

Este script NO forma parte de la aportación principal del TFG. Es un
experimento exploratorio, a mano alzada, motivado por la curiosidad de
comprobar si una arquitectura basada en atención (sin el sesgo inductivo
convolucional de la CNN o la ResNet-18) presenta una SRF distinta. No
pretende ser un estudio riguroso: un ViT está sometido a muchas decisiones
de diseño interno (tamaño de parche, profundidad, número de cabezas de
atención, etc.) que quedan fuera del alcance de este trabajo y que
convendría estudiar con más detalle en un trabajo futuro.

Para mantener el experimento ligero y no introducir más grados de libertad
de los necesarios, se reutiliza exactamente el mismo esquema que para la
ResNet-18 preentrenada (kaggle_vision_resnet18_spectral_cdf_experiment.py):
backbone preentrenado en ImageNet, completamente congelado, con una cabeza
lineal nueva entrenada sobre sus features. El backbone usado es ViT-B/16
(torchvision, pesos ImageNet1k), el mismo tipo de "linear probe" ya usado
para la ResNet-18.

Por defecto se ejecuta solo sobre MNIST, con pocas semillas y un barrido
espectral más corto que en el resto del TFG, precisamente para que el coste
computacional sea bajo (ViT-B/16 a 224x224 es más pesado que ResNet-18).

Uso en Kaggle (requiere GPU para ser viable en un tiempo razonable):
    python kaggle_srf_vit_experiment.py
    python kaggle_srf_vit_experiment.py --datasets mnist fashion_mnist
    python kaggle_srf_vit_experiment.py --quick   # prueba rapida local
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchvision

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Datasets (torchvision trae los 4 de fábrica, incluido KMNIST)
# ---------------------------------------------------------------------------

DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.MNIST},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.FashionMNIST},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.KMNIST},
    "cifar10": {"display_name": "CIFAR-10 (gris)", "image_shape": (32, 32), "torch_cls": torchvision.datasets.CIFAR10},
}
DATASET_ORDER = ["mnist", "fashion_mnist", "kmnist", "cifar10"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

VIT_FEATURE_DIM = 768  # ViT-B/16, hidden_dim en torchvision


def _robust_torchvision_cifar10_download(data_root, max_retries=10):
    """torchvision.datasets.CIFAR10(download=True) descarga cifar-10-python.tar.gz
    desde cs.toronto.edu de una sola vez y sin resume; en Kaggle esa conexión
    se corta con frecuencia a mitad de transferencia, lo que hace que
    download_url() acabe lanzando "File not found or corrupted.". Se rellena
    antes el archivo exacto que espera torchvision con una descarga con
    reintentos y resume, para que su propio download() lo encuentre ya
    válido y no toque nunca la red inestable."""
    import hashlib
    import time
    import urllib.request

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    expected_md5 = "c58f30108f718f92721af3b95e74349a"
    os.makedirs(data_root, exist_ok=True)
    target = os.path.join(data_root, "cifar-10-python.tar.gz")

    def md5_of(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(target) and md5_of(target) == expected_md5:
        return

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = os.path.getsize(target) if os.path.exists(target) else 0
            req = urllib.request.Request(url)
            if resume_from:
                req.add_header("Range", f"bytes={resume_from}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab"
                if resume_from and getattr(resp, "status", 200) != 206:
                    resume_from, mode = 0, "wb"
                with open(target, mode) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            if md5_of(target) == expected_md5:
                return
            print(f"CIFAR-10 archive md5 mismatch after attempt {attempt}/{max_retries}, retrying...")
        except Exception as exc:
            print(f"CIFAR-10 download attempt {attempt}/{max_retries} failed ({exc}); retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Could not download a valid CIFAR-10 archive after {max_retries} attempts.")


def load_dataset_raw(dataset: str, data_root: str):
    spec = DATASET_SPECS[dataset]
    cls = spec["torch_cls"]
    if dataset == "cifar10":
        _robust_torchvision_cifar10_download(data_root)
    train_ds = cls(root=data_root, train=True, download=True)
    test_ds = cls(root=data_root, train=False, download=True)
    if dataset == "cifar10":
        x_train = train_ds.data.astype("float32")
        x_test = test_ds.data.astype("float32")
        y_train = np.array(train_ds.targets, dtype=np.int64)
        y_test = np.array(test_ds.targets, dtype=np.int64)

        def to_gray(x):
            return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]

        x_train, x_test = to_gray(x_train), to_gray(x_test)
    else:
        x_train = train_ds.data.numpy().astype("float32")
        x_test = test_ds.data.numpy().astype("float32")
        y_train = train_ds.targets.numpy().astype(np.int64)
        y_test = test_ds.targets.numpy().astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


@dataclass
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASET_ORDER))
    seeds: list[int] = field(default_factory=lambda: [42])
    n_points: int = 40
    eval_subset_size: int = 500
    val_per_class: int = 200
    train_per_class: int = 1000
    test_per_class: int = 500
    head_epochs: int = 30
    patience: int = 5
    batch_size: int = 128
    feature_batch_size: int = 128
    learning_rate: float = 1e-3
    output_root: str = ""
    data_root: str = ""
    quick: bool = False


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Anexo exploratorio: SRF de un ViT-B/16 preentrenado y congelado (linear probe)."
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_ORDER), choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-points", type=int, default=40)
    parser.add_argument("--eval-subset-size", type=int, default=500, help="-1 para usar el test completo en el barrido")
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignorando argumentos de notebook/kernel: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, n_points=args.n_points, eval_subset_size=args.eval_subset_size,
        val_per_class=args.val_per_class, train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        head_epochs=args.head_epochs, patience=args.patience, batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size, learning_rate=args.learning_rate,
        output_root=args.output_root, data_root=args.data_root, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.n_points = min(cfg.n_points, 10)
        cfg.eval_subset_size = min(cfg.eval_subset_size, 80) if cfg.eval_subset_size > 0 else 80
        cfg.val_per_class = min(cfg.val_per_class, 30)
        cfg.train_per_class = 200 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 200)
        cfg.test_per_class = 50 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 50)
        cfg.head_epochs = min(cfg.head_epochs, 3)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/srf_vit_outputs"
    return "srf_vit_experiment/outputs"


def default_data_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/torch_data"
    return "srf_vit_experiment/torch_data"


def make_output_dir(root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(root or default_output_root()) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_multiclass_subset(x: np.ndarray, y: np.ndarray, limit_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        rng.shuffle(idx)
        effective_limit = len(idx) if limit_per_class <= 0 else min(limit_per_class, len(idx))
        chosen = idx[:effective_limit]
        xs.append(x[chosen])
        ys.append(y[chosen].astype(np.int64))
    x_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(x_out))
    return x_out[perm], y_out[perm]


def load_dataset_three_way(dataset: str, cfg: ExperimentConfig, seed: int) -> dict[str, np.ndarray]:
    data_root = cfg.data_root or default_data_root()
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_dataset_raw(dataset, data_root)
    rng = np.random.default_rng(seed)

    train_idx, val_idx = [], []
    for digit in range(10):
        idx = np.where(y_train_full == digit)[0]
        rng.shuffle(idx)
        n_val = min(cfg.val_per_class, len(idx) - 1)
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    x_train_pool, y_train_pool = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    x_train, y_train = select_multiclass_subset(x_train_pool, y_train_pool, cfg.train_per_class, seed)
    x_test, y_test = select_multiclass_subset(x_test_full, y_test_full, cfg.test_per_class, seed + 1000)

    def prep(x):
        return x.astype("float32") / 255.0

    x_train, x_val, x_test = prep(x_train), prep(x_val), prep(x_test)
    return {
        "x_train_img": x_train, "x_val_img": x_val, "x_test_img": x_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
    }


def subsample_for_sweep(x: np.ndarray, y: np.ndarray, subset_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if subset_size is None or subset_size <= 0 or subset_size >= len(x):
        return x, y
    rng = np.random.default_rng(seed + 2000)
    idx = rng.choice(len(x), size=subset_size, replace=False)
    return x[idx], y[idx]


# ---------------------------------------------------------------------------
# ViT-B/16 congelado + cabeza lineal entrenable
# ---------------------------------------------------------------------------

def build_frozen_vit(device: torch.device) -> nn.Module:
    try:
        backbone = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    except AttributeError:
        backbone = torchvision.models.vit_b_16(pretrained=True)
    backbone.heads = nn.Identity()  # deja el embedding de 768 dims del token [CLS], sin la cabeza de 1000 clases de ImageNet
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    return backbone


class LinearHead(nn.Module):
    def __init__(self, in_features: int = VIT_FEATURE_DIM, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)


def preprocess_for_vit(images_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images_hw)).unsqueeze(1).to(device).float()
    x = Fnn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x


@torch.no_grad()
def extract_features(backbone: nn.Module, images_hw: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    feats = []
    for start in range(0, len(images_hw), batch_size):
        batch = images_hw[start:start + batch_size]
        x = preprocess_for_vit(batch, device)
        feats.append(backbone(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def evaluate_full_pipeline(backbone: nn.Module, head: nn.Module, images_hw: np.ndarray, y: np.ndarray,
                            device: torch.device, batch_size: int) -> float:
    correct = 0
    for start in range(0, len(images_hw), batch_size):
        batch = images_hw[start:start + batch_size]
        x = preprocess_for_vit(batch, device)
        logits = head(backbone(x))
        preds = logits.argmax(dim=1).cpu().numpy()
        correct += int((preds == y[start:start + batch_size]).sum())
    return correct / len(images_hw)


def train_head(head: nn.Module, x_train_feat: np.ndarray, y_train: np.ndarray, x_val_feat: np.ndarray, y_val: np.ndarray,
               device: torch.device, epochs: int, batch_size: int, lr: float, patience: int) -> float:
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    x_train_t = torch.from_numpy(x_train_feat).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    x_val_t = torch.from_numpy(x_val_feat).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    best_val_acc, best_state, stale_epochs = -1.0, None, 0
    n = len(x_train_t)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        running_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = Fnn.cross_entropy(head(x_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_acc = (head(x_val_t).argmax(dim=1) == y_val_t).float().mean().item()
        print(f"    epoch {epoch + 1}/{epochs} - loss={running_loss / n:.4f} - val_accuracy={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_state, stale_epochs = val_acc, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return best_val_acc


# ---------------------------------------------------------------------------
# Barrido espectral acumulativo (idéntica mecánica FFT que en el resto del TFG)
# ---------------------------------------------------------------------------

def radial_frequency_grid_centered(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cy, cx = height // 2, width // 2
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def build_keep_count_schedule(total_modes: int, n_points: int) -> np.ndarray:
    raw = np.linspace(1, total_modes, min(n_points, total_modes))
    counts = np.unique(np.round(raw).astype(int))
    return np.clip(counts, 1, total_modes)


def reconstruct_with_keep_count(images: np.ndarray, order: np.ndarray, height: int, width: int, n_keep: int) -> np.ndarray:
    keep_flat = np.zeros(height * width, dtype=bool)
    keep_flat[order[:n_keep]] = True
    keep_mask = keep_flat.reshape(height, width)
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * keep_mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0)


def sweep_cumulative_lowpass(backbone: nn.Module, head: nn.Module, x_test_img: np.ndarray, y_test: np.ndarray,
                              height: int, width: int, n_points: int, batch_size: int,
                              device: torch.device) -> list[dict[str, float]]:
    radius = radial_frequency_grid_centered(height, width)
    order = np.argsort(radius, axis=None)
    sorted_radius = radius.flatten()[order]
    total_modes = height * width
    rho_max = float(sorted_radius[-1])
    counts = build_keep_count_schedule(total_modes, n_points)

    rows = []
    for n_keep in counts:
        degraded = reconstruct_with_keep_count(x_test_img, order, height, width, int(n_keep))
        accuracy = evaluate_full_pipeline(backbone, head, degraded, y_test, device, batch_size)
        radius_cutoff = float(sorted_radius[n_keep - 1])
        rows.append({
            "modes_kept": int(n_keep), "radius_cutoff": radius_cutoff,
            "normalized_radius": radius_cutoff / rho_max, "accuracy": float(accuracy),
        })
    return rows


def pool_adjacent_violators(y: np.ndarray) -> np.ndarray:
    values: list[float] = []
    weights: list[float] = []
    for yi in y:
        values.append(float(yi))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_value = (values[-2] * weights[-2] + values[-1] * weights[-1]) / merged_weight
            values.pop()
            weights.pop()
            values[-1] = merged_value
            weights[-1] = merged_weight
    fitted = np.empty(len(y), dtype=float)
    pos = 0
    for value, weight in zip(values, weights):
        n = int(round(weight))
        fitted[pos:pos + n] = value
        pos += n
    return fitted


def fit_isotonic_cdf(normalized_accuracy: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(normalized_accuracy, dtype=float), 0.0, 1.0)
    return np.clip(pool_adjacent_violators(y), 0.0, 1.0)


def cdf_statistics(normalized_radius: np.ndarray, cdf_values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(normalized_radius, dtype=float)
    f = np.asarray(cdf_values, dtype=float)
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        f = np.concatenate([[0.0], f])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        f = np.concatenate([f, [1.0]])

    survival = 1.0 - f
    mean = float(_trapz(survival, x))
    second_moment = float(2.0 * _trapz(x * survival, x))
    variance = max(second_moment - mean ** 2, 0.0)
    std = float(np.sqrt(variance))
    median = float(np.interp(0.5, f, x))

    widths_raw = np.diff(x)
    nonzero = widths_raw > 1e-12
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

    auc = float(_trapz(f, x))
    gaussian_entropy = float(0.5 * np.log(2 * np.pi * np.e * max(variance, eps)))
    entropy_gap = gaussian_entropy - entropy

    return {
        "mean": mean, "std": std, "median": median, "mode": mode, "auc": auc,
        "entropy": entropy, "entropy_gap": entropy_gap, "x": x.tolist(), "f": f.tolist(),
    }


# ---------------------------------------------------------------------------
# Ejecución por semilla y por dataset
# ---------------------------------------------------------------------------

def run_seed(dataset: str, cfg: ExperimentConfig, seed: int, backbone: nn.Module, device: torch.device) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]

    print(f"  [seed={seed}] extrayendo features del ViT-B/16 congelado y entrenando cabeza lineal...")
    set_seed(seed)
    data = load_dataset_three_way(dataset, cfg, seed)

    train_feats = extract_features(backbone, data["x_train_img"], device, cfg.feature_batch_size)
    val_feats = extract_features(backbone, data["x_val_img"], device, cfg.feature_batch_size)
    head = LinearHead(VIT_FEATURE_DIM, 10).to(device)
    val_acc = train_head(head, train_feats, data["y_train"], val_feats, data["y_val"], device,
                          cfg.head_epochs, cfg.batch_size, cfg.learning_rate, cfg.patience)

    baseline_acc = evaluate_full_pipeline(backbone, head, data["x_test_img"], data["y_test"], device, cfg.batch_size)
    print(f"  [seed={seed}] val_accuracy={val_acc*100:.2f}% ; accuracy base test completo={baseline_acc*100:.2f}% ; barriendo espectro...")

    x_sweep, y_sweep = subsample_for_sweep(data["x_test_img"], data["y_test"], cfg.eval_subset_size, seed)
    rows = sweep_cumulative_lowpass(backbone, head, x_sweep, y_sweep, height, width, cfg.n_points, cfg.batch_size, device)
    accuracy_full = rows[-1]["accuracy"]
    for row in rows:
        row["normalized_accuracy"] = float(np.clip(row["accuracy"] / accuracy_full, 0.0, 1.0)) if accuracy_full > 0 else 0.0

    x = np.array([r["normalized_radius"] for r in rows])
    y = np.array([r["normalized_accuracy"] for r in rows])
    f_iso = fit_isotonic_cdf(y)
    stats = cdf_statistics(x, f_iso)
    return {"seed": seed, "baseline_accuracy": float(baseline_acc), "rows": rows, "stats": stats}


def run_dataset(dataset: str, cfg: ExperimentConfig, backbone: nn.Module, device: torch.device) -> dict[str, Any]:
    per_seed = [run_seed(dataset, cfg, seed, backbone, device) for seed in cfg.seeds]

    x_common = np.array(per_seed[0]["stats"]["x"])
    f_matrix = np.array([s["stats"]["f"] for s in per_seed])
    f_mean = f_matrix.mean(axis=0)
    f_std = f_matrix.std(axis=0)

    agg_stats: dict[str, dict[str, float]] = {}
    for key in ("mean", "std", "median", "mode", "auc", "entropy", "entropy_gap"):
        values = [s["stats"][key] for s in per_seed]
        agg_stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    baseline_accs = [s["baseline_accuracy"] for s in per_seed]
    agg_stats["baseline_accuracy"] = {"mean": float(np.mean(baseline_accs)), "std": float(np.std(baseline_accs)), "values": baseline_accs}

    return {
        "dataset": dataset, "display_name": DATASET_SPECS[dataset]["display_name"],
        "eval_subset_size": cfg.eval_subset_size,
        "per_seed": per_seed, "x_common": x_common.tolist(), "f_mean": f_mean.tolist(), "f_std": f_std.tolist(),
        "aggregate_stats": agg_stats,
    }


# ---------------------------------------------------------------------------
# Figuras y reporte
# ---------------------------------------------------------------------------

def save_vit_plot(result: dict[str, Any], output_path: Path) -> None:
    x_common = np.array(result["x_common"])
    f_mean = np.array(result["f_mean"])
    f_std = np.array(result["f_std"])
    plt.figure(figsize=(7, 5))
    for idx, entry in enumerate(result["per_seed"]):
        plt.plot(entry["stats"]["x"], entry["stats"]["f"], color="C0", alpha=0.3, linewidth=1.0,
                  label=f"curvas individuales ({len(result['per_seed'])} semillas)" if idx == 0 else None)
    plt.fill_between(x_common, np.clip(f_mean - f_std, 0, 1), np.clip(f_mean + f_std, 0, 1),
                      color="C1", alpha=0.25, label="banda ± 1 desviación típica")
    plt.plot(x_common, f_mean, color="C1", linewidth=2.5, label="curva media")
    plt.xlabel("Radio espectral normalizado (r = ρ/ρ_max)")
    plt.ylabel("Precisión normalizada (F(r))")
    plt.title(f"SRF exploratoria — ViT-B/16 congelado — {result['display_name']}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Anexo exploratorio: SRF de un ViT-B/16 preentrenado y congelado (linear probe)",
        "",
        "**Aviso:** este experimento no forma parte del análisis comparativo riguroso de los "
        "Capítulos 2 y 3. Es un estudio exploratorio, a mano alzada, motivado por la curiosidad "
        "de comprobar si una arquitectura basada en atención se comporta de forma distinta a la "
        "CNN y a la ResNet-18. No pretende ser un análisis exhaustivo de los muchos grados de "
        "libertad de diseño de un ViT (tamaño de parche, profundidad, número de cabezas, etc.); "
        "se deja como avance de trabajo futuro.",
        "",
        "## Setup",
        f"- Semillas: `{cfg.seeds}`",
        f"- Puntos por curva: `{cfg.n_points}`",
        f"- Subconjunto de test para el barrido: `{cfg.eval_subset_size if cfg.eval_subset_size > 0 else 'completo'}`",
        "",
        "| Dataset | Accuracy base | E[R] | σ[R] | Mediana | Moda | AUC | H | ΔH |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        agg = result["aggregate_stats"]
        lines.append(
            f"| {result['display_name']} "
            f"| {agg['baseline_accuracy']['mean']*100:.2f}% ± {agg['baseline_accuracy']['std']*100:.2f}pp "
            f"| {agg['mean']['mean']:.4f} ± {agg['mean']['std']:.4f} "
            f"| {agg['std']['mean']:.4f} ± {agg['std']['std']:.4f} "
            f"| {agg['median']['mean']:.4f} ± {agg['median']['std']:.4f} "
            f"| {agg['mode']['mean']:.4f} ± {agg['mode']['std']:.4f} "
            f"| {agg['auc']['mean']:.4f} ± {agg['auc']['std']:.4f} "
            f"| {agg['entropy']['mean']:.4f} ± {agg['entropy']['std']:.4f} "
            f"| {agg['entropy_gap']['mean']:.4f} ± {agg['entropy_gap']['std']:.4f} |"
        )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    output_dir = make_output_dir(cfg.output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            _ = torch.zeros(1, device=device) + 1.0
        except RuntimeError as exc:
            print(f"AVISO: se detectó GPU pero no es compatible con este build de PyTorch ({exc}). "
                  f"En Kaggle esto pasa con el acelerador 'GPU P100' (arquitectura Pascal, sm_60): el "
                  f"PyTorch preinstalado solo soporta sm_70 en adelante. Cambia el acelerador del notebook "
                  f"a 'GPU T4 x2' (Settings -> Accelerator) y vuelve a ejecutar. Cayendo a CPU por ahora "
                  f"(será muy lento; usa --quick para una prueba).")
            device = torch.device("cpu")
    if device.type == "cpu":
        print("AVISO: ejecutando en CPU. ViT-B/16 a 224x224 será muy lento; usa --quick para una prueba rápida.")
    print(f"Dispositivo: {device}")
    print(f"Conjuntos de datos: {cfg.datasets}  |  Semillas: {cfg.seeds}")
    print(f"Directorio de salida: {output_dir.resolve()}")

    backbone = build_frozen_vit(device)

    results = []
    for dataset in cfg.datasets:
        print(f"\n===== DATASET: {DATASET_SPECS[dataset]['display_name']} =====")
        result = run_dataset(dataset, cfg, backbone, device)
        results.append(result)
        save_vit_plot(result, output_dir / f"{dataset}_vit_srf.png")
        with (output_dir / f"{dataset}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    build_report(output_dir, cfg, results)
    with (output_dir / "metrics_all_datasets.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(cfg), "results": results}, fh, indent=2)

    print("\n=== Resumen agregado ===")
    for result in results:
        agg = result["aggregate_stats"]
        print(
            f"{result['display_name']}: accuracy_base={agg['baseline_accuracy']['mean']*100:.2f}%±{agg['baseline_accuracy']['std']*100:.2f}pp "
            f"E[R]={agg['mean']['mean']:.4f}±{agg['mean']['std']:.4f} "
            f"AUC={agg['auc']['mean']:.4f}±{agg['auc']['std']:.4f} "
            f"ΔH={agg['entropy_gap']['mean']:.4f}±{agg['entropy_gap']['std']:.4f}"
        )
    print(f"Resultados guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
