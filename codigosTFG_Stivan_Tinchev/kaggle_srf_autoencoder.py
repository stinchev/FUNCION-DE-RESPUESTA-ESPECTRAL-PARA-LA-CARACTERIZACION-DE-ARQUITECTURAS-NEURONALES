from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

_trapz = getattr(np, "trapezoid", None) or np.trapz

def load_mnist_raw():
    return keras.datasets.mnist.load_data()

def load_fashion_mnist_raw():
    return keras.datasets.fashion_mnist.load_data()

def _load_kmnist_from_npz():
    base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
    files = {
        "x_train": "kmnist-train-imgs.npz", "y_train": "kmnist-train-labels.npz",
        "x_test": "kmnist-test-imgs.npz", "y_test": "kmnist-test-labels.npz",
    }
    arrays = {}
    for key, fname in files.items():
        path = keras.utils.get_file(fname, origin=base_url + fname, cache_subdir="datasets/kmnist")
        arrays[key] = np.load(path)["arr_0"]
    return (arrays["x_train"], arrays["y_train"]), (arrays["x_test"], arrays["y_test"])

def load_kmnist_raw():
    try:
        import tensorflow_datasets as tfds
        x_train, y_train = tfds.as_numpy(tfds.load("kmnist", split="train", as_supervised=True, batch_size=-1))
        x_test, y_test = tfds.as_numpy(tfds.load("kmnist", split="test", as_supervised=True, batch_size=-1))
        return (x_train.squeeze(-1), y_train), (x_test.squeeze(-1), y_test)
    except Exception as exc:
        print(f"tensorflow_datasets no disponible para KMNIST ({exc}); descargando .npz oficiales...")
        return _load_kmnist_from_npz()

def _robust_cifar10_download(max_retries=10):
    import hashlib
    import tempfile
    import time
    import urllib.request

    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    expected_sha256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"

    if "KERAS_HOME" in os.environ:
        base = os.path.expanduser(os.environ["KERAS_HOME"])
    else:
        base = os.path.expanduser("~/.keras")
    if not os.path.isdir(base) or not os.access(base, os.W_OK):
        base = os.path.join(tempfile.gettempdir(), ".keras")
    datadir = os.path.join(base, "datasets")
    os.makedirs(datadir, exist_ok=True)
    target = os.path.join(datadir, "cifar-10-batches-py-target_archive")

    def sha256_of(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(target) and sha256_of(target) == expected_sha256:
        return

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = os.path.getsize(target) if os.path.exists(target) else 0
            req = urllib.request.Request(origin)
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
            if sha256_of(target) == expected_sha256:
                return
            print(f"CIFAR-10 archive hash mismatch after attempt {attempt}/{max_retries}, retrying...")
        except Exception as exc:
            print(f"CIFAR-10 download attempt {attempt}/{max_retries} failed ({exc}); retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Could not download a valid CIFAR-10 archive after {max_retries} attempts.")

def robust_cifar10_load_data():
    _robust_cifar10_download()
    return keras.datasets.cifar10.load_data()

def load_cifar10_grayscale_raw():
    (x_train, y_train), (x_test, y_test) = robust_cifar10_load_data()
    y_train, y_test = y_train.squeeze(-1), y_test.squeeze(-1)

    def to_gray(x):
        x = x.astype("float32")
        return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]

    return (to_gray(x_train), y_train), (to_gray(x_test), y_test)

DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "loader": load_mnist_raw},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "loader": load_fashion_mnist_raw},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "loader": load_kmnist_raw},
    "cifar10": {"display_name": "CIFAR-10 (gris)", "image_shape": (32, 32), "loader": load_cifar10_grayscale_raw},
}
DATASET_ORDER = ["mnist", "fashion_mnist", "kmnist", "cifar10"]

@dataclass
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASET_ORDER))
    seeds: list[int] = field(default_factory=lambda: [42])
    latent_dim: int = 32
    n_points: int = 60
    eval_subset_size: int = 1500
    val_per_class: int = 300
    train_per_class: int = 2000
    test_per_class: int = 500
    ae_epochs: int = 25
    ae_patience: int = 5
    head_epochs: int = 20
    head_patience: int = 4
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_root: str = ""
    quick: bool = False

def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Anexo exploratorio: SRF de un autoencoder usado para clasificacion en su espacio latente."
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_ORDER), choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-points", type=int, default=60)
    parser.add_argument("--eval-subset-size", type=int, default=1500, help="-1 para usar el test completo en el barrido")
    parser.add_argument("--val-per-class", type=int, default=300)
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--ae-epochs", type=int, default=25)
    parser.add_argument("--ae-patience", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--head-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignorando argumentos de notebook/kernel: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, latent_dim=args.latent_dim, n_points=args.n_points,
        eval_subset_size=args.eval_subset_size,
        val_per_class=args.val_per_class, train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        ae_epochs=args.ae_epochs, ae_patience=args.ae_patience, head_epochs=args.head_epochs,
        head_patience=args.head_patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
        output_root=args.output_root, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.n_points = min(cfg.n_points, 15)
        cfg.eval_subset_size = min(cfg.eval_subset_size, 100) if cfg.eval_subset_size > 0 else 100
        cfg.val_per_class = min(cfg.val_per_class, 50)
        cfg.train_per_class = 300 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 300)
        cfg.test_per_class = 100 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 100)
        cfg.ae_epochs = min(cfg.ae_epochs, 3)
        cfg.head_epochs = min(cfg.head_epochs, 3)
    return cfg

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/srf_autoencoder_outputs"
    return "srf_autoencoder_experiment/outputs"

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
        ys.append(y[chosen].astype(np.int32))
    x_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(x_out))
    return x_out[perm], y_out[perm]

def load_dataset_three_way(dataset: str, cfg: ExperimentConfig, seed: int) -> dict[str, np.ndarray]:
    (x_train_full, y_train_full), (x_test_full, y_test_full) = DATASET_SPECS[dataset]["loader"]()
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
        "y_train": y_train.astype("int32"), "y_val": y_val.astype("int32"), "y_test": y_test.astype("int32"),
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
    }

def subsample_for_sweep(x: np.ndarray, y: np.ndarray, subset_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if subset_size is None or subset_size <= 0 or subset_size >= len(x):
        return x, y
    rng = np.random.default_rng(seed + 2000)
    idx = rng.choice(len(x), size=subset_size, replace=False)
    return x[idx], y[idx]

def build_autoencoder(image_shape: tuple[int, int], latent_dim: int, learning_rate: float):
    height, width = image_shape
    h4, w4 = height // 4, width // 4

    encoder_input = keras.layers.Input(shape=(height, width, 1))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(encoder_input)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(latent_dim, name="latent")(x)
    encoder = keras.Model(encoder_input, latent, name="ae_encoder")

    decoder_input = keras.layers.Input(shape=(latent_dim,))
    y = keras.layers.Dense(h4 * w4 * 128, activation="relu")(decoder_input)
    y = keras.layers.Reshape((h4, w4, 128))(y)
    y = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(y)
    y = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(y)
    reconstruction = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(y)
    decoder = keras.Model(decoder_input, reconstruction, name="ae_decoder")

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name="autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return encoder, decoder, autoencoder

def fit_autoencoder(autoencoder, x_train, x_val, epochs, patience, batch_size):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-5),
    ]
    autoencoder.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=epochs,
                     batch_size=batch_size, verbose=2, callbacks=callbacks)

def build_and_fit_linear_head(encoder, x_train, y_train, x_val, y_val, epochs, patience, batch_size, learning_rate):
    encoder.trainable = False
    inputs = keras.layers.Input(shape=encoder.input_shape[1:])
    logits = keras.layers.Dense(10, activation="softmax")(encoder(inputs, training=False))
    classifier = keras.Model(inputs, logits, name="ae_linear_probe")
    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True)]
    classifier.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                    batch_size=batch_size, verbose=2, callbacks=callbacks)
    return classifier

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

def sweep_cumulative_lowpass(classifier, x_test_img: np.ndarray, y_test: np.ndarray, height: int, width: int,
                              n_points: int, batch_size: int) -> list[dict[str, float]]:
    radius = radial_frequency_grid_centered(height, width)
    order = np.argsort(radius, axis=None)
    sorted_radius = radius.flatten()[order]
    total_modes = height * width
    rho_max = float(sorted_radius[-1])
    counts = build_keep_count_schedule(total_modes, n_points)

    rows = []
    for n_keep in counts:
        degraded = reconstruct_with_keep_count(x_test_img, order, height, width, int(n_keep))
        _, accuracy = classifier.evaluate(degraded[..., np.newaxis], y_test, batch_size=batch_size, verbose=0)
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

def run_seed(dataset: str, cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]

    print(f"  [seed={seed}] entrenando autoencoder (reconstrucción)...")
    set_seed(seed)
    data = load_dataset_three_way(dataset, cfg, seed)
    x_train = data["x_train_img"][..., np.newaxis]
    x_val = data["x_val_img"][..., np.newaxis]
    x_test = data["x_test_img"][..., np.newaxis]

    encoder, decoder, autoencoder = build_autoencoder((height, width), cfg.latent_dim, cfg.learning_rate)
    fit_autoencoder(autoencoder, x_train, x_val, cfg.ae_epochs, cfg.ae_patience, cfg.batch_size)
    reconstruction_mse = float(autoencoder.evaluate(x_test, x_test, verbose=0))
    print(f"  [seed={seed}] MSE de reconstrucción en test = {reconstruction_mse:.5f} ; entrenando clasificador lineal sobre z...")

    classifier = build_and_fit_linear_head(
        encoder, x_train, data["y_train"], x_val, data["y_val"],
        cfg.head_epochs, cfg.head_patience, cfg.batch_size, cfg.learning_rate,
    )
    _, baseline_acc = classifier.evaluate(x_test, data["y_test"], verbose=0)
    print(f"  [seed={seed}] accuracy base (clasificador lineal sobre z, imagen sin tocar) = {baseline_acc*100:.2f}% ; barriendo espectro...")

    x_sweep, y_sweep = subsample_for_sweep(data["x_test_img"], data["y_test"], cfg.eval_subset_size, seed)
    rows = sweep_cumulative_lowpass(classifier, x_sweep, y_sweep, height, width, cfg.n_points, cfg.batch_size)
    accuracy_full = rows[-1]["accuracy"]
    for row in rows:
        row["normalized_accuracy"] = float(np.clip(row["accuracy"] / accuracy_full, 0.0, 1.0)) if accuracy_full > 0 else 0.0

    x = np.array([r["normalized_radius"] for r in rows])
    y = np.array([r["normalized_accuracy"] for r in rows])
    f_iso = fit_isotonic_cdf(y)
    stats = cdf_statistics(x, f_iso)
    return {
        "seed": seed, "baseline_accuracy": float(baseline_acc),
        "reconstruction_mse": reconstruction_mse, "rows": rows, "stats": stats,
    }

def run_dataset(dataset: str, cfg: ExperimentConfig) -> dict[str, Any]:
    per_seed = [run_seed(dataset, cfg, seed) for seed in cfg.seeds]

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
    recon_mses = [s["reconstruction_mse"] for s in per_seed]
    agg_stats["reconstruction_mse"] = {"mean": float(np.mean(recon_mses)), "std": float(np.std(recon_mses)), "values": recon_mses}

    return {
        "dataset": dataset, "display_name": DATASET_SPECS[dataset]["display_name"], "latent_dim": cfg.latent_dim,
        "per_seed": per_seed, "x_common": x_common.tolist(), "f_mean": f_mean.tolist(), "f_std": f_std.tolist(),
        "aggregate_stats": agg_stats,
    }

def save_ae_plot(result: dict[str, Any], output_path: Path) -> None:
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
    plt.title(f"SRF exploratoria — Autoencoder (z∈R^{result['latent_dim']}) — {result['display_name']}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Anexo exploratorio: SRF de un autoencoder (clasificación en espacio latente)",
        "",
        "**Aviso:** este experimento no forma parte del análisis comparativo riguroso de los "
        "Capítulos 2 y 3. Es un estudio exploratorio, a mano alzada, para comprobar si la SRF "
        "de una arquitectura de naturaleza distinta (con un objetivo de reconstrucción y una "
        "dimensión de espacio latente que no existen en la CNN ni en la ResNet-18) se comporta "
        "de forma distinta. Se deja como avance de trabajo futuro, no como una conclusión firme.",
        "",
        "## Setup",
        f"- Semillas: `{cfg.seeds}`",
        f"- Dimensión del espacio latente: `{cfg.latent_dim}`",
        f"- Puntos por curva: `{cfg.n_points}`",
        f"- Subconjunto de test para el barrido: `{cfg.eval_subset_size if cfg.eval_subset_size > 0 else 'completo'}`",
        "",
        "| Dataset | Accuracy base | MSE reconstrucción | E[R] | σ[R] | Mediana | Moda | AUC | H | ΔH |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        agg = result["aggregate_stats"]
        lines.append(
            f"| {result['display_name']} "
            f"| {agg['baseline_accuracy']['mean']*100:.2f}% ± {agg['baseline_accuracy']['std']*100:.2f}pp "
            f"| {agg['reconstruction_mse']['mean']:.5f} ± {agg['reconstruction_mse']['std']:.5f} "
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
    print(f"Conjuntos de datos: {cfg.datasets}  |  Semillas: {cfg.seeds}  |  latent_dim={cfg.latent_dim}")
    print(f"Directorio de salida: {output_dir.resolve()}")

    results = []
    for dataset in cfg.datasets:
        print(f"\n===== DATASET: {DATASET_SPECS[dataset]['display_name']} =====")
        result = run_dataset(dataset, cfg)
        results.append(result)
        save_ae_plot(result, output_dir / f"{dataset}_ae_srf.png")
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
            f"MSE_recon={agg['reconstruction_mse']['mean']:.5f} "
            f"E[R]={agg['mean']['mean']:.4f}±{agg['mean']['std']:.4f} "
            f"AUC={agg['auc']['mean']:.4f}±{agg['auc']['std']:.4f} "
            f"ΔH={agg['entropy_gap']['mean']:.4f}±{agg['entropy_gap']['std']:.4f}"
        )
    print(f"Resultados guardados en: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
