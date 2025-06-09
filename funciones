import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Definir la función original
def f(x):
    return x  # Función ejemplo: f(x) = x en [-π, π]

# Intervalo del período
L = np.pi

# Número de términos de la serie
N = 10

# Coeficientes a_0, a_n y b_n
def a0():
    result, _ = quad(lambda x: f(x), -L, L)
    return result / (2 * L)

def an(n):
    result, _ = quad(lambda x: f(x) * np.cos(n * np.pi * x / L), -L, L)
    return result / L

def bn(n):
    result, _ = quad(lambda x: f(x) * np.sin(n * np.pi * x / L), -L, L)
    return result / L

# Función aproximada por la serie de Fourier
def fourier_series(x, N):
    sum = a0()
    for n in range(1, N + 1):
        sum += an(n) * np.cos(n * np.pi * x / L) + bn(n) * np.sin(n * np.pi * x / L)
    return sum

# Graficar la función original y su aproximación
x_vals = np.linspace(-L, L, 1000)
f_vals = np.vectorize(f)(x_vals)
f_fourier_vals = np.vectorize(lambda x: fourier_series(x, N))(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label="Función original f(x) = x", color='blue')
plt.plot(x_vals, f_fourier_vals, label=f"Serie de Fourier con N={N}", color='red', linestyle='--')
plt.title("Aproximación de la Serie de Fourier")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()
