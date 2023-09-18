import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cs

seed = 59529
system = SolarSystem(seed)

num_planets = system.number_of_planets

with open('positions.txt', 'r') as infile:
    lines = infile.readlines()
    pos = np.zeros((num_planets, len(lines), 2))
    for j in range(len(lines)):
        data = lines[j].strip('\n').split(',')
        for i in range(num_planets):
            x, y = data[i].split(';')
            pos[i,j] = float(x), float(y)


a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
M_s = system.star_mass              # Massen til Stellaris Skarsgard
M_zeron = system.masses[0]          # Massen til Zeron

# Perioden til Zeron: bruker denne til a finne tiden det tar for 30 omlop rundt Stel. Skars.
period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(cs.G_sol*(M_s+M_zeron)))
dt = 1e-4
t = np.arange(0,30*period_zeron,dt)

N = len(pos[0])
dA = np.zeros(N)
for i in range(N-1):
    dA[i] = np.linalg.norm(np.cross(pos[0,i], pos[0,i+1]))/2

plt.plot(t[:-2], dA[:-1])
plt.show()

