from __future__ import annotations

import argparse
import json
import os
import random
from collections import OrderedDict
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

DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.MNIST},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.FashionMNIST},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.KMNIST},
    "cifar10": {"display_name": "CIFAR-10 (gris)", "image_shape": (32, 32), "torch_cls": torchvision.datasets.CIFAR10},
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _robust_torchvision_cifar10_download(data_root, max_retries=10):
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
        except Exception as exc:
            print(f"CIFAR-10 download attempt {attempt}/{max_retries} failed ({exc}); retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Could not download a valid CIFAR-10 archive after {max_retries} attempts.")

def load_dataset_raw(dataset: str, data_root: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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
    datasets: list[str] = field(default_factory=lambda: ["mnist"])
    seeds: list[int] = field(default_factory=lambda: [42])
    n_points: int = 150
    eval_subset_size: int = 1000
    val_per_class: int = 200
    train_per_class: int = 1000
    test_per_class: int = -1
    epochs: int = 10
    patience: int = 3
    batch_size: int = 64
    head_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-5
    output_root: str = ""
    data_root: str = ""
    quick: bool = False

def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Descongelado progresivo del backbone + SRF, ResNet-18.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mnist"], choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-points", type=int, default=150)
    parser.add_argument("--eval-subset-size", type=int, default=1000, help="-1 para usar el test completo en el barrido")
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--train-per-class", type=int, default=1000, help="-1 para usar todo el conjunto de entrenamiento")
    parser.add_argument("--test-per-class", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, n_points=args.n_points,
        eval_subset_size=args.eval_subset_size, val_per_class=args.val_per_class, train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        head_learning_rate=args.head_learning_rate, backbone_learning_rate=args.backbone_learning_rate,
        output_root=args.output_root, data_root=args.data_root, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.n_points = min(cfg.n_points, 10)
        cfg.eval_subset_size = 50
        cfg.val_per_class = 20
        cfg.train_per_class = 100
        cfg.test_per_class = 50
        cfg.epochs = 2
    return cfg

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/progressive_unfreezing_resnet18_outputs"
    return "progressive_unfreezing_resnet18_experiment/outputs"

def default_data_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/torch_data"
    return "progressive_unfreezing_experiment/torch_data"

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

class LinearHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)

def build_backbone_and_plan(device: torch.device) -> tuple[nn.Module, int, "OrderedDict[str, list[nn.Module]]"]:
    try:
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        backbone = torchvision.models.resnet18(pretrained=True)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    block_map: "OrderedDict[str, list[nn.Module]]" = OrderedDict([
        ("layer4.1", [backbone.layer4[1]]),
        ("layer4.0", [backbone.layer4[0]]),
        ("layer3.1", [backbone.layer3[1]]),
        ("layer3.0", [backbone.layer3[0]]),
        ("layer2.1", [backbone.layer2[1]]),
        ("layer2.0", [backbone.layer2[0]]),
        ("layer1.1", [backbone.layer1[1]]),
        ("layer1.0", [backbone.layer1[0]]),
        ("stem(conv1+bn1)", [backbone.conv1, backbone.bn1]),
    ])

    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    return backbone, feature_dim, block_map

def apply_stage(backbone: nn.Module, block_map: "OrderedDict[str, list[nn.Module]]", stage_idx: int) -> list[str]:
    for p in backbone.parameters():
        p.requires_grad_(False)
    unfrozen_names = list(block_map.keys())[:stage_idx]
    for name in unfrozen_names:
        for module in block_map[name]:
            for p in module.parameters():
                p.requires_grad_(True)
    return unfrozen_names

def set_backbone_train_mode(backbone: nn.Module, block_map: "OrderedDict[str, list[nn.Module]]", unfrozen_names: list[str]) -> None:
    backbone.eval()
    for name in unfrozen_names:
        for module in block_map[name]:
            module.train()

def fraction_unfrozen(backbone: nn.Module, total_backbone_params: int) -> float:
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return trainable / total_backbone_params if total_backbone_params > 0 else 0.0

def preprocess_for_backbone(images_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images_hw)).unsqueeze(1).to(device).float()
    x = Fnn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x

@torch.no_grad()
def evaluate_full_pipeline(backbone: nn.Module, head: nn.Module, images_hw: np.ndarray, y: np.ndarray,
                            device: torch.device, batch_size: int) -> float:
    backbone.eval()
    head.eval()
    correct = 0
    for start in range(0, len(images_hw), batch_size):
        batch = images_hw[start:start + batch_size]
        x = preprocess_for_backbone(batch, device)
        logits = head(backbone(x))
        preds = logits.argmax(dim=1).cpu().numpy()
        correct += int((preds == y[start:start + batch_size]).sum())
    return correct / len(images_hw)

def train_full_pipeline(backbone: nn.Module, head: nn.Module, block_map: "OrderedDict[str, list[nn.Module]]",
                         unfrozen_names: list[str], x_train_img: np.ndarray, y_train: np.ndarray,
                         x_val_img: np.ndarray, y_val: np.ndarray, device: torch.device, cfg: ExperimentConfig) -> float:
    trainable_backbone_params = [p for name in unfrozen_names for module in block_map[name]
                                  for p in module.parameters() if p.requires_grad]
    param_groups = [{"params": head.parameters(), "lr": cfg.head_learning_rate}]
    if trainable_backbone_params:
        param_groups.append({"params": trainable_backbone_params, "lr": cfg.backbone_learning_rate})
    optimizer = torch.optim.Adam(param_groups)

    n = len(x_train_img)
    best_val_acc, best_backbone_state, best_head_state, stale = -1.0, None, None, 0
    for epoch in range(cfg.epochs):
        set_backbone_train_mode(backbone, block_map, unfrozen_names)
        head.train()
        perm = np.random.permutation(n)
        running_loss = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            x = preprocess_for_backbone(x_train_img[idx], device)
            y = torch.from_numpy(y_train[idx]).long().to(device)
            optimizer.zero_grad()
            loss = Fnn.cross_entropy(head(backbone(x)), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(idx)
        val_acc = evaluate_full_pipeline(backbone, head, x_val_img, y_val, device, cfg.batch_size)
        print(f"      epoch {epoch + 1}/{cfg.epochs} - loss={running_loss / n:.4f} - val_accuracy={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_backbone_state = {k: v.clone() for k, v in backbone.state_dict().items()}
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_backbone_state is not None:
        backbone.load_state_dict(best_backbone_state)
        head.load_state_dict(best_head_state)
    return best_val_acc

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

def discrete_stats_with_entropy(x: np.ndarray, srf: np.ndarray) -> dict[str, float]:
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

def run_stage(stage_idx: int, unfrozen_names: list[str], dataset: str, cfg: ExperimentConfig,
              device: torch.device) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]

    per_seed_rows, per_seed_stats, baseline_accs = [], [], []
    for seed in cfg.seeds:
        set_seed(seed)
        backbone, feature_dim, block_map = build_backbone_and_plan(device)
        apply_stage(backbone, block_map, stage_idx)
        head = LinearHead(feature_dim, 10).to(device)

        data = load_dataset_three_way(dataset, cfg, seed)
        print(f"    [seed={seed}] stage {stage_idx} ({', '.join(unfrozen_names) if unfrozen_names else 'solo cabeza'}) - entrenando...")
        val_acc = train_full_pipeline(backbone, head, block_map, unfrozen_names,
                                       data["x_train_img"], data["y_train"], data["x_val_img"], data["y_val"],
                                       device, cfg)
        baseline_acc = evaluate_full_pipeline(backbone, head, data["x_test_img"], data["y_test"], device, cfg.batch_size)
        print(f"    [seed={seed}] val_accuracy={val_acc * 100:.2f}% - accuracy base (test completo)={baseline_acc * 100:.2f}% - barriendo espectro...")

        x_sweep, y_sweep = subsample_for_sweep(data["x_test_img"], data["y_test"], cfg.eval_subset_size, seed)
        rows = sweep_cumulative_lowpass(backbone, head, x_sweep, y_sweep, height, width, cfg.n_points, cfg.batch_size, device)

        x, srf = srf_from_rows(rows)
        stats = discrete_stats_with_entropy(x, srf)

        per_seed_rows.append(rows)
        per_seed_stats.append(stats)
        baseline_accs.append(float(baseline_acc))

    agg_stats: dict[str, dict[str, float]] = {}
    for key in ("mean", "std", "median", "mode", "H"):
        values = [s[key] for s in per_seed_stats]
        agg_stats[key] = {"media": float(np.mean(values)), "std": float(np.std(values)), "valores": values}
    agg_stats["baseline_accuracy"] = {
        "media": float(np.mean(baseline_accs)), "std": float(np.std(baseline_accs)), "valores": baseline_accs
    }

    x0, srf0 = srf_from_rows(per_seed_rows[0])
    all_srf = [srf_from_rows(rows)[1] for rows in per_seed_rows]
    mean_srf = np.mean(np.stack(all_srf, axis=0), axis=0)

    return {
        "stage": stage_idx, "unfrozen_blocks": unfrozen_names,
        "mean_normalized_radius": x0.tolist(), "mean_srf": mean_srf.tolist(),
        "aggregate_stats": agg_stats,
        "per_seed_rows": per_seed_rows,
    }

def run_experiment(dataset: str, cfg: ExperimentConfig, device: torch.device) -> list[dict[str, Any]]:
    _, _, block_map_template = build_backbone_and_plan(device)
    n_units = len(block_map_template)
    del block_map_template

    results = []
    for stage_idx in range(0, n_units + 1):
        backbone_probe, _, block_map_probe = build_backbone_and_plan(device)
        total_backbone_params = sum(p.numel() for p in backbone_probe.parameters())
        unfrozen_names = apply_stage(backbone_probe, block_map_probe, stage_idx)
        frac = fraction_unfrozen(backbone_probe, total_backbone_params)
        del backbone_probe, block_map_probe

        print(f"\n  --- Stage {stage_idx}/{n_units} (resnet18, {DATASET_SPECS[dataset]['display_name']}): "
              f"descongelado {frac * 100:.1f}% del backbone ({', '.join(unfrozen_names) if unfrozen_names else 'ninguno'}) ---")
        result = run_stage(stage_idx, unfrozen_names, dataset, cfg, device)
        result["fraccion_backbone_descongelado"] = frac
        results.append(result)
    return results

def build_srs_table(dataset: str, results: list[dict[str, Any]]) -> str:
    lines = [
        f"# Tabla de SRS por etapa de descongelado -- resnet18 -- {DATASET_SPECS[dataset]['display_name']}",
        "",
        "| Stage | Bloques descongelados | % backbone descongelado | Precisión base | E[R] | σ[R] | Mediana | Moda | H |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        agg = r["aggregate_stats"]
        blocks = ", ".join(r["unfrozen_blocks"]) if r["unfrozen_blocks"] else "(solo cabeza)"
        lines.append(
            f"| {r['stage']} | {blocks} | {r['fraccion_backbone_descongelado'] * 100:.1f}% "
            f"| {agg['baseline_accuracy']['media'] * 100:.2f}%±{agg['baseline_accuracy']['std'] * 100:.2f}pp "
            f"| {agg['mean']['media']:.4f}±{agg['mean']['std']:.4f} "
            f"| {agg['std']['media']:.4f}±{agg['std']['std']:.4f} "
            f"| {agg['median']['media']:.4f}±{agg['median']['std']:.4f} "
            f"| {agg['mode']['media']:.4f}±{agg['mode']['std']:.4f} "
            f"| {agg['H']['media']:.4f}±{agg['H']['std']:.4f} |"
        )
    return "\n".join(lines)

def build_srf_long_table(results: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for r in results:
        for x_val, srf_val in zip(r["mean_normalized_radius"], r["mean_srf"]):
            rows.append({"stage": r["stage"], "fraccion_backbone_descongelado": r["fraccion_backbone_descongelado"],
                         "normalized_radius": x_val, "srf": srf_val})
    return rows

def save_srf_long_csv(rows: list[dict[str, float]], path: Path) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "fraccion_backbone_descongelado", "normalized_radius", "srf"])
        writer.writeheader()
        writer.writerows(rows)

def save_comparison_plot(dataset: str, results: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for i, r in enumerate(results):
        plt.plot(r["mean_normalized_radius"], r["mean_srf"], color=f"C{i}", label=f"stage {r['stage']}")
    plt.xlabel("Radio espectral normalizado (r = ρ/ρ_max)")
    plt.ylabel("SRF(r)")
    plt.title("SRF por etapa de descongelado")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def main() -> None:
    cfg = parse_args()
    output_dir = make_output_dir(cfg.output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            _ = torch.zeros(1, device=device) + 1.0
        except RuntimeError as exc:
            print(f"AVISO: GPU detectada pero incompatible con PyTorch ({exc}). ")
            device = torch.device("cpu")
    if device.type == "cpu":
        print("AVISO: ejecutando en CPU.")
    print(f"Device: {device}")
    print("Arquitectura: resnet18")
    print(f"Datasets: {cfg.datasets}")
    print(f"Output directory: {output_dir.resolve()}")

    for dataset in cfg.datasets:
        print(f"\n===== ARQUITECTURA: resnet18 -- DATASET: {DATASET_SPECS[dataset]['display_name']} =====")
        results = run_experiment(dataset, cfg, device)

        with (output_dir / f"{dataset}_all_stages_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump({"config": asdict(cfg), "results": results}, fh, indent=2)

        srs_table_md = build_srs_table(dataset, results)
        (output_dir / f"{dataset}_srs_table.md").write_text(srs_table_md, encoding="utf-8")
        print("\n" + srs_table_md)

        srf_long_rows = build_srf_long_table(results)
        save_srf_long_csv(srf_long_rows, output_dir / f"{dataset}_srf_all_stages_long.csv")

        save_comparison_plot(dataset, results, output_dir / f"{dataset}_srf_por_etapa.png")

    print(f"\nGuardado en: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
