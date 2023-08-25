# seed er 59529
import numpy as np
# import scipy.stats as st
import scipy.constants as cs
import random as rand
from ast2000tools.solar_system import SolarSystem

# Initialbetingelser
# Lengdene paa boksen i x-, y- og z-retning
box = np.asarray([1e-6, 1e-6, 1e-6])  # meters
edge_tolerance = 1e-9
# Lengdene paa hullet i x- og y-retning
sx, sy = 0.5*box[0], 0.5*box[1]
T = 3e3  # kelvin
N = 100  # number of particles
system = SolarSystem(59529)
particle_mass = 0.1  # kg

pos = np.zeros((N, 3))
vel = np.zeros((N, 3))

# trekker tilfeldige startposisjoner og starthastigheter
# fra relevante fordelinger
for i in range(N):
    for j in range(3):
        pos[i, j] = rand.uniform(0, box[j])
        vel[i, j] = rand.gauss(0, np.sqrt(cs.k*T/particle_mass))

dt = 1e-12
t = 0

while t < 1e-9:
    for i in range(N):
        for j in range(3):
            pos[i, j] = pos[i, j] + vel[i, j]*dt

            if pos[i, j] < edge_tolerance or pos[i, j] > box[j]-edge_tolerance:
                vel[i, j] = - vel[i, j]
    t += dt

for i in range(N):
    for j in range(3):
        assert (0 < pos[i, j] < box[j])
