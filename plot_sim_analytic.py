#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.solar_system import SolarSystem
from plotting_orbits import main

seed = 59529
system = SolarSystem(seed)

num_planets = system.number_of_planets

pos = np.load('positions.npy')

plt.style.use('dark_background')
main()

colors = ['#e79797', '#fabd55', '#fae155', \
            '#55fa8f', '#83d2ff', '#cec1ff', '#f4c1ff']

for i in range(num_planets):
    plt.plot(pos[i,:,0], pos[i,:,1], ':', color=colors[i], alpha=0.7, label=f'sim planet {i}')
    plt.scatter(pos[i,-1,0], pos[i,-1,1], color=colors[i])

plt.axis('equal')
plt.legend()
plt.show()