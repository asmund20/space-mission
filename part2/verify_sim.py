#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

import numpy as np
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cs

seed = 59529
system = SolarSystem(seed)

num_planets = system.number_of_planets

pos = np.load('positions.npy')

a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
M_s = system.star_mass              # Massen til Stellaris Skarsgard
M_zeron = system.masses[0]          # Massen til Zeron
period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(cs.G_sol*(M_s+M_zeron)))

system.verify_planet_positions(30*period_zeron, pos)