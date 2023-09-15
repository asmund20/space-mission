import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.solar_system import SolarSystem
from plotting_orbits import main

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

plt.style.use('dark_background')
main()

for i in range(num_planets):
    plt.plot(pos[i,::5,0], pos[i,::5,1], ':', label=f'sim planet {i}')

plt.axis('equal')
plt.legend()
plt.show()