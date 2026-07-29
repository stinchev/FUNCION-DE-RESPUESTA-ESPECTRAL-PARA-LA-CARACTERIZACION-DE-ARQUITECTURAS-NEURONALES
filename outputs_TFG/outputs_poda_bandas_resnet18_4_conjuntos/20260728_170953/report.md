# Poda de bandas con ResNet-18 congelada (barrido progresivo + eliminación completa), 4 datasets

## Setup
- Seeds: `[42, 43, 44]`
- Paso del barrido progresivo: `5%`
- Subconjunto de test para el barrido progresivo: `2000`

## MNIST
- Accuracy base (test completo): `97.76% ± 0.05pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 261 | 16.88% ± 0.37pp | 80.88pp |
| Media | 268 | 29.47% ± 0.54pp | 68.29pp |
| Alta | 255 | 82.64% ± 1.20pp | 15.12pp |

## Fashion-MNIST
- Accuracy base (test completo): `89.18% ± 0.12pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 261 | 29.22% ± 0.78pp | 59.96pp |
| Media | 268 | 53.45% ± 0.39pp | 35.72pp |
| Alta | 255 | 77.09% ± 0.82pp | 12.08pp |

## KMNIST
- Accuracy base (test completo): `83.87% ± 0.55pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 261 | 13.10% ± 0.14pp | 70.76pp |
| Media | 268 | 42.42% ± 0.33pp | 41.45pp |
| Alta | 255 | 70.72% ± 0.85pp | 13.14pp |

## CIFAR-10 (gris)
- Accuracy base (test completo): `79.25% ± 0.18pp`

| Banda eliminada | Modos eliminados | Precisión | Caída |
|---|---|---|---|
| Baja | 341 | 14.69% ± 0.32pp | 64.57pp |
| Media | 340 | 52.13% ± 0.17pp | 27.13pp |
| Alta | 343 | 73.10% ± 0.08pp | 6.16pp |
