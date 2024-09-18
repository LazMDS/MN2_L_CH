import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import pandas as pd

print("1D heat equation solver")

comprimento_da_placa = 271 #mm
tempo_maximo = 600

alpha = 97 # mm^2/seg (convertendo de mm^2/min para mm^2/seg)
delta_x = 10


delta_t = ((delta_x ** 2)/(4 * alpha))
gamma = (alpha * delta_t) / (delta_x ** 2)

num_points = int(comprimento_da_placa / delta_x)

u = np.empty((tempo_maximo, num_points))
u_inicial = 30.0

u_top = 1000.0
u_botton = 0.0

u[0, :] = u_inicial

u[0, 0] = u_top
u[0, -1] = u_botton

def calculate(u):
    for k in range(0, tempo_maximo-1, 1):
        if k * delta_t >= 300:
            u_top = 0
        else:
            u_top = 1000

        for i in range(1, num_points-1, 1):
            u[k + 1, i] = gamma * (u[k, i+1]+ u[k,i-1] - 2*u[k, i]) + u[k,i]

        # Manter as condições de contorno fixas
        u[k + 1, 0] = u_top
        u[k + 1, -1] = u_botton

    return u

def plotheatmap(u_k, k):
  plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
  plt.xlabel("x")
  plt.ylabel("y")
  
  # Gráfico 2D representando a variação da temperatura ao longo do tempo (em cores)
  plt.imshow([u_k], cmap='hot', aspect='auto', vmin=0, vmax=1000)
  plt.colorbar(label='Temperatura (°C)')
  
  return plt

u = calculate(u)

times = np.arange(0, tempo_maximo, 1)
positions = np.arange(0, num_points, 1)

assert len(times) == u.shape[0], "O comprimento do índice não corresponde ao número de linhas em u"
assert len(positions) == u.shape[1], "O comprimento das posições não corresponde ao número de colunas em u"

data = {f"Position {pos:.1f} mm": u[:, i] for i, pos in enumerate(positions)}
df = pd.DataFrame(data, index=times)

df.to_csv("heat_equation_solution.csv", float_format='%.2f')

print("Data saved to heat_equation_solution.csv")

def animate(k):
    # Clear the current plot figure
    plt.clf()
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, frames=tempo_maximo, interval=50, repeat=False)
anim.save("heat_equation_solution.gif")

print("Done!")