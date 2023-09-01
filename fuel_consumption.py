import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as st
import scipy.constants as cs
import random as rand
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import ast2000tools

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
mr = SpaceMission(seed).spacecraft_mass
# Masse til Zeron
Mz = system.masses[0]*ast2000tools.constants.m_sun
# Radius Zeron
Rz = system.radii[0]*1e3

def simulate_launch(N, fuel_mass, n_boxes, consume_fuel=True):
    # fuel_consumtion, thrust = np.array(microbox_performance(N))*n_boxes
    fuel_consumtion, thrust = np.array((1.13544e-15, 9.899740313191332e-12))*n_boxes

    # Max tid for aa oppnaa unnslipningshastighet er 20 min
    t_max = 1200 # s
    dt = 1e-3 # s

    # Vertikal posisjon
    z = np.zeros(int(t_max/dt))
    z[0] = Rz

    # Fart i vertikal retning
    vz = np.zeros(int(t_max/dt))

    # Akselerasjonen
    az = np.zeros(int(t_max/dt))

    # Masse
    mass = np.zeros(int(t_max/dt))
    mass[0] = mr + fuel_mass
    fuel = np.zeros(int(t_max/dt))
    fuel[0] = fuel_mass - n_boxes*N*particle_mass

    esc_vel = np.zeros(int(t_max/dt))

    
    t = 0
    i = 0
    while i < int(t_max/dt)-1:

        esc_vel[i] = np.sqrt((2*cs.G*Mz)/z[i])
        gravity = -cs.G*mass[i]*Mz/(z[i]**2)
        az[i] = (thrust + gravity)/mass[i]
        if vz[i] >= esc_vel[i]:
            break

        vz[i+1] = vz[i] + az[i]*dt
        z[i+1] = z[i] + vz[i+1]*dt

        mass[i+1] = mass[i] - fuel_consumtion*dt
        fuel[i+1] = fuel[i] - fuel_consumtion*dt

        t += dt
        i += 1
    
    return t, z[:i+1], vz[:i+1], az[:i+1], mass[:i+1], fuel[:i+1], esc_vel[:i+1]

t, z, vz, az, mass, fuel, esc_vel = simulate_launch(1000, 12000, 2.7e16)
print(fuel[-1])
time = np.linspace(0,t,len(z))

fig, axs = plt.subplots(3)
fig.suptitle('Simulering av rakettoppskytning', fontweight='bold')

axs[0].plot(time, z, 'r-', label='Avstand fra sentrum til raketten')
axs[0].set_ylabel('Avstand [m]')
axs[0].set_xlabel('Tid [s]')
axs[0].legend()

axs[1].plot(time, vz, 'g-', label='Fart til raketten')
axs[1].plot(time, esc_vel, 'k:', label='Unnsplipningshastighet')
axs[1].set_ylabel('Fart [m/s]')
axs[1].set_xlabel('Tid [s]')
axs[1].legend()

axs[2].plot(time, az, 'b-', label='Akselerasjonen til raketten')
axs[2].set_ylabel('Akselerasjon [m/s^2]')
axs[2].set_xlabel('Tid [s]')
axs[2].legend()


plt.tight_layout()
plt.show()

