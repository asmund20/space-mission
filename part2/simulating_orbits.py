#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

import numpy as np
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as cs
from numba import jit 

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
M_s = system.star_mass              # Massen til Stellaris Skarsgard
M_zeron = system.masses[0]          # Massen til Zeron

# Perioden til Zeron: bruker denne til a finne tiden det tar for 30 omlop rundt Stel. Skars.
period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(cs.G_sol*(M_s+M_zeron)))
t_max = 30*period_zeron
dt = 1e-4   # 10,000 tidssteg per yr
N = int(t_max/dt)   # Antall tidssteg

@jit(nopython=True)
def integrator(pos, vel, a):
    j = 0
    while j < N-1:
        for i in range(num_planets):

            # Her bruker vi Leap Frog-metoden for numerisk integrasjon
            pos[i,j+1] = pos[i,j] + vel[i,j]*dt + 0.5*a[i,j]*(dt**2)
            a[i,j+1] =  -cs.G_sol*M_s*pos[i,j+1]/(np.linalg.norm(pos[i,j+1])**3)
            vel[i,j+1] = vel[i,j] + 0.5*(a[i,j]+a[i,j+1])*dt
    
        j += 1
    return pos, vel, a


num_planets = system.number_of_planets
pos = np.zeros((num_planets, N, 2))
vel = np.zeros((num_planets, N, 2))
a = np.zeros((num_planets, N, 2))

for i in range(num_planets):
    pos[i,0] = system.initial_positions[0,i], system.initial_positions[1,i]
    vel[i,0] = system.initial_velocities[0,i], system.initial_velocities[1,i]
    a[i,0] = -cs.G_sol*M_s*pos[i,0]/(np.linalg.norm(pos[i,0])**3)

pos, vel, a = integrator(pos, vel, a)


with open('positions.txt', 'w') as pos_outfile:
    for j in range(N):
        str = f''
        for i in range(num_planets):
            str += f'{pos[i,j,0]};{pos[i,j,1]},'
        pos_outfile.write(str+'\n')

t = np.linspace(0, t_max, N)
pos = np.reshape(pos, (2, num_planets, N))
mission.generate_orbit_video(t, pos)
