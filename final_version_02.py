import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress
import pandas as pd
from tqdm import trange

# --------------------------
# PARÁMETROS DEL SISTEMA
# --------------------------
N = 200                  # Número de partículas
L = 10.0                 # Tamaño de la caja (cúbica)
T_total = 10.0           # Tiempo total de simulación
dt = 0.01                # Paso de tiempo
steps = int(T_total / dt)
m = 1.0                  # Masa de cada partícula

# Distribuciones
tipos = ['gaussiana', 'uniforme', 'pareto']
colores = {'gaussiana': 'blue', 'uniforme': 'green', 'pareto': 'red'}

# --------------------------
# FUNCIONES AUXILIARES
# --------------------------
def init_positions(N, L):
    return np.random.uniform(0, L, size=(N, 3))

def init_velocities(N, tipo='gaussiana'):
    if tipo == 'gaussiana':
        V = np.random.normal(loc=0.0, scale=1.0, size=(N, 3))
    elif tipo == 'uniforme':
        V = np.random.uniform(-1.0, 1.0, size=(N, 3))
    elif tipo == 'pareto':
        V = (np.random.pareto(2.5, size=(N, 3)) + 1) * np.random.choice([-1, 1], size=(N, 3))
    else:
        raise ValueError("Distribución no reconocida")
    return V

def apply_wall_collisions(R, V, L):
    for dim in range(3):
        out_left = R[:, dim] < 0
        out_right = R[:, dim] > L
        V[out_left | out_right, dim] *= -1
        R[:, dim] = np.clip(R[:, dim], 0, L)
    return R, V

# --------------------------
# CONFIGURACIÓN INICIAL
# --------------------------
simulaciones = {}

for tipo in tipos:
    R = init_positions(N, L)
    V = init_velocities(N, tipo)
    R0 = R.copy()
    simulaciones[tipo] = {
        'R': R,
        'V': V,
        'R0': R0,
        'MSD': [],
        'times': [],
    }

# --------------------------
# ANIMACIÓN EN TIEMPO REAL
# --------------------------
fig = plt.figure(figsize=(15, 5))
axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]
scatters = []

for ax, tipo in zip(axes, tipos):
    R = simulaciones[tipo]['R']
    scat = ax.scatter(R[:, 0], R[:, 1], R[:, 2], color=colores[tipo], s=10)
    ax.set_title(tipo.capitalize())
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    scatters.append(scat)

def update(frame):
    for i, tipo in enumerate(tipos):
        sim = simulaciones[tipo]
        R, V = sim['R'], sim['V']
        R += V * dt
        R, V = apply_wall_collisions(R, V, L)
        sim['R'], sim['V'] = R, V

        if frame % 10 == 0:
            dr2 = np.sum((R - sim['R0'])**2, axis=1)
            sim['MSD'].append(np.mean(dr2))
            sim['times'].append(frame * dt)

        scatters[i]._offsets3d = (R[:, 0], R[:, 1], R[:, 2])
    return scatters

ani = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
ani.save("difusion_multidistrib.gif", writer="pillow", fps=30)
plt.show()

# --------------------------
# GRAFICAR SOLO LAS CURVAS MSD
# --------------------------
plt.figure(figsize=(10, 6))
for tipo in tipos:
    t, msd = simulaciones[tipo]['times'], simulaciones[tipo]['MSD']
    plt.plot(t, msd, label=f'{tipo}', color=colores[tipo], linewidth=2)

    # Guardar datos CSV
    df = pd.DataFrame({'tiempo': t, 'MSD': msd})
    df.to_csv(f'msd_{tipo}.csv', index=False)

    # Coeficiente de difusión
    slope, intercept, r_value, p_value, std_err = linregress(t, msd)
    D = slope / 6
    print(f"{tipo.title():<10} | D ≈ {D:.4f} | R² = {r_value**2:.4f}")

plt.xlabel('Tiempo')
plt.ylabel('Desplazamiento cuadrático medio (MSD)')
plt.title('MSD vs Tiempo (solo curvas)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('curvas_MSD_comparadas.png', dpi=300)
plt.show()
