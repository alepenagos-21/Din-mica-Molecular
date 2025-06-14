#  Simulación de Difusión en 3D con Diferentes Distribuciones Iniciales de Velocidad

Este proyecto realiza una simulación de difusión de partículas dentro de una caja tridimensional, evaluando el efecto de tres tipos de distribuciones iniciales de velocidad:

- Distribución **gaussiana**
- Distribución **uniforme**
- Distribución **de Pareto**

El objetivo es visualizar y comparar el comportamiento difusivo resultante, utilizando herramientas de animación y análisis numérico en Python.

---

## Contenidos del repositorio

| Archivo | Descripción |
|--------|-------------|
| `final_version.py` | Script principal con animación 3D y análisis comparativo del MSD. |
| `difusion_multidistrib.gif` | Animación que muestra las tres distribuciones en tiempo real. |
| `curvas_MSD_comparadas.png` | Gráfica final comparando el MSD entre las tres distribuciones. |
| `msd_gaussiana.csv`, `msd_uniforme.csv`, `msd_pareto.csv` | Datos numéricos del MSD para cada distribución. |

---

##  Requisitos

Este proyecto requiere Python 3.7+ y las siguientes bibliotecas:

```bash
pip install numpy==1.26.4 scipy matplotlib pandas tqdm pillow

