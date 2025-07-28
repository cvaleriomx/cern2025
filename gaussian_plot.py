import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Parámetro sigma
sigma = 2.0

# Rango de x
x = np.linspace(-3 * sigma, 3 * sigma, 500)

# Gaussiana normalizada con desviación estándar sigma
gauss = 4*(1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x**2 / (2 * sigma**2))

# Función error
erf_values = erf(x / (sigma * np.sqrt(2)))  # ajustar x para que se relacione con sigma



# Función error
erf_valuesL = 0.5*(-erf(x / (sigma * np.sqrt(2)))+1)
erf_valuesR = 0.5*(erf(x / (sigma * np.sqrt(2)))+1)


# Crear figura con dos ejes
fig, axs = plt.subplots()

# Plot de la gaussiana
axs.plot(x, gauss, label=fr'Gaussiana ($\sigma$={sigma})', color='blue')
axs.plot(x, erf_values, label=r'$\mathrm{erf}(x)$', color='red')

#axs[1].set_title("Función Error")
axs.grid(True)
axs.legend()

plt.xlabel("x")
plt.tight_layout()
fig2, axs2 = plt.subplots()
axs2.plot(x, gauss, label=r'$\frac{2}{\sqrt{\pi}} e^{-x^2}$', color='green')

axs2.plot(x, erf_valuesL, label=r'0.5*($\mathrm{erf}(x)+1$) Left')
axs2.plot(x, erf_valuesR, label=r'0.5*($\mathrm{erf}(x)+1$) Right')
axs2.legend(loc='upper right')

fig3, axs3 = plt.subplots()
axs3.plot(x, gauss, label=r'$\frac{2}{\sqrt{\pi}} e^{-x^2}$', color='green')

#erf_valuesL = -0.5*(-erf(x)+1)+0.5
#erf_valuesR = 0.5*(erf(x)+1)-0.5

xL = np.linspace(-3 * sigma, 0, 500)
xR = np.linspace(0, 3 * sigma, 500)

# Función error
erf_valuesL = 2*(0.5*(erf(xL/ (sigma * np.sqrt(2)))+1)-0.5)
erf_valuesR = 2*(0.5*(erf(xR / (sigma * np.sqrt(2)))+1)-0.5)


axs3.plot(xL, erf_valuesL, label=r'0.5*($\mathrm{erf}(x)+1$) Left')
axs3.plot(xR, erf_valuesR, label=r'0.5*($\mathrm{erf}(x)+1$) Right')
axs3.legend(loc='upper right')
axs3.grid(True)

plt.show()
