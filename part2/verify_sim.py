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

N = len(pos[0,:])
pos = np.reshape(pos, (2, num_planets, N))

a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
M_s = system.star_mass              # Massen til Stellaris Skarsgard
M_zeron = system.masses[0]          # Massen til Zeron
period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(cs.G_sol*(M_s+M_zeron)))

system.verify_planet_positions(30*period_zeron, pos)