import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as st
import scipy.constants as cs
import random as rand
from ast2000tools.solar_system import SolarSystem

seed = 59529

rand.seed(seed)

# Initialbetingelser
# Lengdene paa boksen i x-, y- og z-retning
box = np.asarray([1e-6, 1e-6, 1e-6])  # meters
edge_tolerance = 2e-9
# radius paa hullet
r = box[0]/2/np.sqrt(np.pi)
T = 3e3  # kelvin
N = 10000  # number of particles
#N = 1000  # for raskere kjøring ved jobb
system = SolarSystem(seed)
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
dt = 1e-11
t = 0
RUNTIME = 1e-8

test_x = list()
t_list = list()
total_impulse_particles = 0
N_particles_escaped = 0
total_impulse_escaped_particles = 0

while t < RUNTIME:
    for i in range(N):
        for j in range(3):
            pos[i, j] = pos[i, j] + vel[i, j]*dt

            if pos[i, j] < edge_tolerance and j == 2 and np.linalg.norm(
                    np.asarray([box[0]/2, box[1]/2, pos[i, j]])-pos[i]) < r:
                N_particles_escaped += 1
                total_impulse_escaped_particles -= 2*vel[i, j]*particle_mass

            if pos[i, j] < edge_tolerance or \
                    pos[i, j] > box[j]-edge_tolerance:
                total_impulse_particles += 2*abs(vel[i, j])*particle_mass
                vel[i, j] = - vel[i, j]

    test_x.append(pos[1, 0])
    t_list.append(t)

    t += dt

plt.plot(t_list, test_x)
plt.show()
A = 2*(box[0]*box[1]+box[0]*box[2]+box[1]*box[2])
print(f"pressure_numerical = {total_impulse_particles/RUNTIME/A}\npressure_analytical = {N/box[0]/box[1]/box[2]*cs.k*T}")

total_E = 0
total_velocity = 0
for v in vel:
    total_velocity += np.linalg.norm(v)
    total_E += 1/2*particle_mass*np.linalg.norm(v)**2

print(f"average kinetic energy numerical: {total_E/N}\naverage kinetic energy analytical: {3/2*cs.k*T}")
print(f"average velocity numerical: {total_velocity/N}\naverage velocity analytical: {np.sqrt(8*cs.k*T/np.pi/particle_mass)}")

print(f"fuel comsumption: {N_particles_escaped*particle_mass/RUNTIME} kg/s")
print(f"thrust generated: {total_impulse_escaped_particles/RUNTIME} N")
print(f"thrust = P*A_hull: {total_impulse_particles/RUNTIME/A*0.25*box[0]*box[1]}")
print("Her kan vi sjekke kraft generert analytisk ved hjelp av trykket og arealet til hullet")
