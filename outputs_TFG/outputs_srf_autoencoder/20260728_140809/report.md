# Anexo exploratorio: SRF de un autoencoder (clasificación en espacio latente)

**Aviso:** este experimento no forma parte del análisis comparativo riguroso de los Capítulos 2 y 3. Es un estudio exploratorio, a mano alzada, para comprobar si la SRF de una arquitectura de naturaleza distinta (con un objetivo de reconstrucción y una dimensión de espacio latente que no existen en la CNN ni en la ResNet-18) se comporta de forma distinta. Se deja como avance de trabajo futuro, no como una conclusión firme.

## Setup
- Semillas: `[42]`
- Dimensión del espacio latente: `32`
- Puntos por curva: `60`
- Subconjunto de test para el barrido: `1500`

| Dataset | Accuracy base | MSE reconstrucción | E[R] | σ[R] | Mediana | Moda | AUC | H | ΔH |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | 92.80% ± 0.00pp | 0.00376 ± 0.00000 | 0.1088 ± 0.0000 | 0.0581 ± 0.0000 | 0.1223 ± 0.0000 | 0.1322 ± 0.0000 | 0.8912 ± 0.0000 | -1.5840 ± 0.0000 | 0.1576 ± 0.0000 |
| Fashion-MNIST | 81.74% ± 0.00pp | 0.00767 ± 0.00000 | 0.1109 ± 0.0000 | 0.1056 ± 0.0000 | 0.0844 ± 0.0000 | 0.0565 ± 0.0000 | 0.8891 ± 0.0000 | -1.2271 ± 0.0000 | 0.3976 ± 0.0000 |
| KMNIST | 69.10% ± 0.00pp | 0.02214 ± 0.00000 | 0.1039 ± 0.0000 | 0.0799 ± 0.0000 | 0.0982 ± 0.0000 | 0.1322 ± 0.0000 | 0.8961 ± 0.0000 | -1.2660 ± 0.0000 | 0.1576 ± 0.0000 |
| CIFAR-10 (gris) | 28.98% ± 0.00pp | 0.00955 ± 0.00000 | 0.0396 ± 0.0000 | 0.0505 ± 0.0000 | 0.0143 ± 0.0000 | 0.0494 ± 0.0000 | 0.9604 ± 0.0000 | -1.9978 ± 0.0000 | 0.4300 ± 0.0000 |
