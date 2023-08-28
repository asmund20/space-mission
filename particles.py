# seed er 59529
import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as st
import scipy.constants as cs
import random as rand
from ast2000tools.solar_system import SolarSystem

rand.seed(59529)

# Initialbetingelser
# Lengdene paa boksen i x-, y- og z-retning
box = np.asarray([1e-6, 1e-6, 1e-6])  # meters
edge_tolerance = 2e-9
# Lengdene paa hullet i x- og y-retning
sx, sy = 0.5*box[0], 0.5*box[1]
T = 3e3  # kelvin
N = 100  # number of particles
system = SolarSystem(59529)
particle_mass = 3.32e-27  # kg

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
RUNTIME = 1e-8

test_x = list()
t_list = list()
total_impulse_particles = 0

while t < RUNTIME:
    for i in range(N):
        for j in range(3):
            pos[i, j] = pos[i, j] + vel[i, j]*dt

            if pos[i, j] < edge_tolerance or pos[i, j] > box[j]-edge_tolerance:
                total_impulse_particles += abs(vel[i, j])*particle_mass
                vel[i, j] = - vel[i, j]

    test_x.append(pos[1, 0])
    t_list.append(t)

    t += dt

plt.plot(t_list, test_x)
plt.show()
print(f"pressure_numerical = {total_impulse_particles/RUNTIME/2/(box[0]*box[1]+box[0]*box[2]+box[1]*box[2])}\npressure_analytical = {N/box[0]/box[1]/box[2]*cs.k*T}")

E = 0
for v in vel:
    E += 1/2*particle_mass*np.linalg.norm(v)**2
print(f"total kinetic energy numerical: {E}\ntotal kinetic energy analytical: {3/2*cs.k*T*N}")
