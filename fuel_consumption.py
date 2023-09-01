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
N = 1000  # for raskere kjøring ved jobb
system = SolarSystem(seed)
particle_mass = 3.32e-27  # kg


# Denne funksjonen beregner hvor mye drivstoff som forbrennes
def fuel_consumed(F, consumption, m, dv) -> float:
    return consumption*m*dv/F


def microbox_performance(N):
    """Finner thrust og drivstoff brent av hver mikroboks"""
    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))

    # trekker tilfeldige startposisjoner og starthastigheter
    # fra relevante fordelinger
    for i in range(N):
        for j in range(3):
            pos[i, j] = rand.uniform(0, box[j])
            vel[i, j] = rand.gauss(0, np.sqrt(cs.k*T/particle_mass))

    dt = 1e-12

    total_impulse_particles = 0
    N_particles_escaped = 0
    total_impulse_escaped_particles = 0

    t = 0
    RUNTIME = 1e-8
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
        t += dt
    
    fuel_consumtion = N_particles_escaped*particle_mass/RUNTIME
    thrust = total_impulse_escaped_particles/RUNTIME

    return fuel_consumtion, thrust

# Rakettmasse
mr = 6e4
# Masse til Zeron
Mz = system.masses[0]*ast2000tools.constants.m_sun
# Radius Zeron
Rz = system.radii[0]*1e3

# Tyngdekraft paa overflaten av Zeron
gravity_surface_zeron = cs.G*mr*Mz/(Rz**2)

def simulate_launch(N, fuel_mass, n_boxes, consume_fuel=True):
    fuel_consumtion, thrust = microbox_performance(N)*n_boxes

    total_mass = mr + fuel_mass
    

print(microbox_performance(N))
