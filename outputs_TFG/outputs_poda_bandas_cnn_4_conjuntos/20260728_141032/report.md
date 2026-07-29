# Poda de bandas de la CNN (barrido progresivo + eliminación completa), 4 datasets

## Setup
- Seeds: `[42, 43, 44]`
- Paso del barrido progresivo: `5%`

## MNIST
- Accuracy base: `99.32% ± 0.07pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 261 | 34.72% ± 5.94pp | 64.60pp |
| Media | 268 | 99.27% ± 0.04pp | 0.05pp |
| Alta | 255 | 99.34% ± 0.06pp | -0.02pp |

## Fashion-MNIST
- Accuracy base: `92.55% ± 0.36pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 261 | 34.98% ± 7.18pp | 57.57pp |
| Media | 268 | 90.44% ± 0.28pp | 2.11pp |
| Alta | 255 | 91.93% ± 0.34pp | 0.62pp |

## KMNIST
- Accuracy base: `96.69% ± 0.19pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 261 | 43.52% ± 11.77pp | 53.17pp |
| Media | 268 | 96.52% ± 0.26pp | 0.17pp |
| Alta | 255 | 96.64% ± 0.20pp | 0.04pp |

## CIFAR-10 (gris)
- Accuracy base: `70.68% ± 1.42pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 341 | 11.42% ± 1.75pp | 59.26pp |
| Media | 340 | 67.14% ± 1.55pp | 3.54pp |
| Alta | 343 | 70.14% ± 1.44pp | 0.54pp |
