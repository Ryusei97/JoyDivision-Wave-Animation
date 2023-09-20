from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from math import exp, pi, sin
import random
from tqdm import tqdm


data = []
# Change number of points to change smoothness of the graph
xs = np.linspace(-50, 150, 100)

def R(_n):
    random.seed(_n)
    return random.random()


for m in tqdm(range(1, 81, 1)):
    lines = []
    for t in np.linspace(0, (6.3*18)/19, 18):
        l = []
        for x in xs:
            small = [4*sin(2*pi*R(4*m)+t+R(2*n*m)*2*pi)*exp(-(0.3*x+30-100*R(2*n*m))**2.0/20.0) for n in range(1, 31, 1)]
            large = [3*(1+R(3*n*m))*abs(sin(t+R(n*m)*2*pi))*exp(-(x-100*R(n*m))**2.0/20.0) for n in range(1, 5, 1)]
            d = 80 - m + 0.2*sin(2*pi*R(6*m) + sum(small)) + sum(large)
            l.append(d)
        lines.append(l)
    data.append(lines)

 
# subplots() function you can draw
# multiple plots in one figure
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 8), facecolor='black')
 
# style for plotting line
plt.style.use("ggplot")
 
def animate(i):

    axes.clear()
    axes.set_ylim(-5, 90)
    axes.set_xlim(-60, 160)
    axes.axis('off')

    zorder = 0
    for wave in data:
        axes.plot(xs, wave[i], color='white', zorder=zorder)
        axes.fill_between(xs, wave[i], color='black', alpha=1.0, zorder=zorder)
        zorder += 1

    


anim = FuncAnimation(fig, animate, frames=18, blit=False, interval=100)

# plt.show()
anim.save('sample_wave.gif', writer = 'pillow', fps = 12)