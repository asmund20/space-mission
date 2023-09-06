import numpy as np
import scipy.constants as cs
import random as rand
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as constants
import matplotlib.pyplot as plt


seed = 59529

rand.seed(seed)

system = SolarSystem(seed)
particle_mass = 3.32e-27  # kg


# Denne funksjonen beregner hvor mye drivstoff som forbrennes
def fuel_consumed(F, consumption, m, dv) -> float:
    return consumption*m*dv/F


# finner kraft generert og drivstoffforbruk i kg/s
def microbox_performance(N):
    # Initialbetingelser
    # Lengdene paa boksen i x-, y- og z-retning
    box = np.asarray([1e-6, 1e-6, 1e-6])  # meters
    edge_tolerance = 2e-9
    # radius paa hullet, utledet fra arealet til en sirkel der
    # A = 0.25 L^2
    r = box[0]/2/np.sqrt(np.pi)
    T = 3e3  # kelvin

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

                # Sjekker om partikkelen havnet ut av hullet
                if pos[i, j] < edge_tolerance and j == 2 and np.linalg.norm(
                        np.asarray([box[0]/2, box[1]/2, pos[i, j]])-pos[i]) < r:
                    N_particles_escaped += 1
                    total_impulse_escaped_particles -= 2*vel[i, j]*particle_mass

                # Sjekker om partikkelen spratt i veggen
                if pos[i, j] < edge_tolerance or \
                        pos[i, j] > box[j]-edge_tolerance:
                    total_impulse_particles += 2*abs(vel[i, j])*particle_mass
                    vel[i, j] = - vel[i, j]
        t += dt

    fuel_consumtion = N_particles_escaped*particle_mass/RUNTIME
    thrust = total_impulse_escaped_particles/RUNTIME

    return fuel_consumtion, thrust


def simulate_launch(N, fuel_mass, n_boxes, consume_fuel=True):
    # Rakettmasse
    mr = SpaceMission(seed).spacecraft_mass
    # Masse til Zeron
    Mz = system.masses[0]*constants.m_sun
    # Radius Zeron
    Rz = system.radii[0]*1e3

    fuel_consumption, thrust = np.array(microbox_performance(N))*n_boxes

    # Max tid for aa oppnaa unnslipningshastighet er 20 min
    t_max = 1200  # s
    dt = 1e-3  # s

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

    # Unnsplipningshastighet
    esc_vel = np.zeros(int(t_max/dt))

    t = 0
    i = 0
    while i < int(t_max/dt)-1:

        # oppdaterer unnslipningshastighet, da den er avhengig av
        # posisjonen og massen til raketten
        esc_vel[i] = np.sqrt((2*cs.G*Mz)/z[i])
        # opddaterer kraften på raketten, da den er avhengig av
        # posisjonen og massen
        gravity = -cs.G*mass[i]*Mz/(z[i]**2)
        # opddaterer aksellerasjonen på raketten, da den er avhengig av
        # kraften og massen
        az[i] = (thrust + gravity)/mass[i]

        # avslutter hvis vi har oppnådd unnslipningshastighet
        if vz[i] >= esc_vel[i]:
            break

        # oppdaterer hastighet og posisjon
        vz[i+1] = vz[i] + az[i]*dt
        z[i+1] = z[i] + vz[i+1]*dt

        # oppddaterer rakettens totale masse og mengdden drivstoff
        mass[i+1] = mass[i] - fuel_consumption*dt
        fuel[i+1] = fuel[i] - fuel_consumption*dt

        t += dt
        i += 1

    return t, z[:i+1], vz[:i+1], az[:i+1], mass[:i+1], fuel[:i+1], esc_vel[:i+1], fuel_consumption


t, z, vz, az, mass, fuel, esc_vel, fuel_consumption = simulate_launch(1000, 12000, 2.7e16)
with open('rocket_position.txt', 'w') as outfile:
    for pos in z:
        outfile.write(f'{pos}\n')

time = np.linspace(0, t, len(z))

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

plt.figure()
plt.plot(time, mass, color='orange', label=f"forbruk: {fuel_consumption:.1f} kg/s")
plt.ylabel("drivstoff [kg]")
plt.xlabel("tid [s]")
plt.suptitle(f"Drivstoff i tanken\ngjenværende drivstoff: {fuel[-1]:.1f} kg")
plt.legend()

plt.tight_layout()
plt.show()
