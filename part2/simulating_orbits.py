import numpy as np
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cs
import matplotlib.pyplot as plt

seed = 59529
system = SolarSystem(seed)


a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
M_s = system.star_mass              # Massen til Stellaris Skarsgard
M_zeron = system.masses[0]          # Massen til Zeron

# Perioden til Zeron: bruker denne til a finne tiden det tar for 20 omlop rundt Stel. Skars.
period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(cs.G_sol*(M_s+M_zeron)))
t_max = 30*period_zeron
dt = 0.5e-4   # 20,000 tidssteg per yr

N = int(t_max/dt)   # Antall tidssteg
num_planets = system.number_of_planets
pos = np.zeros((num_planets, N, 2))
vel = np.zeros((num_planets, N, 2))
a = np.zeros((num_planets, N, 2))

for i in range(num_planets):
    pos[i,0] = system.initial_positions[0,i], system.initial_positions[1,i]
    vel[i,0] = system.initial_velocities[0,i], system.initial_velocities[1,i]
    a[i,0] = -cs.G_sol*M_s*pos[i,0]/(np.linalg.norm(pos[i,0])**3)

j = 0
while j < N-1:

    for i in range(num_planets):
        # Her bruker vi Leap Frog-metoden for numerisk integrasjon
        # a[i,j+1] =  -cs.G_sol*M_s*pos[i,j]/(np.linalg.norm(pos[i,j])**3)
        # pos[i,j+1] = pos[i,j] + vel[i,j]*dt + 0.5*a[i,j]*(dt**2)
        # vel[i,j+1] = vel[i,j] + 0.5*(a[i,j]+a[i,j+1])*dt

        a[i,j] =  -cs.G_sol*M_s*pos[i,j]/(np.linalg.norm(pos[i,j])**3)
        vel[i,j+1] = vel[i,j] + a[i,j]*dt
        pos[i,j+1] = pos[i,j] + vel[i,j+1]*dt
    
    j += 1


# with open('positions.txt', 'w') as pos_outfile:
#     for j in range(N):
#         str = f''
#         for i in range(num_planets):
#             str += f'{pos[i,j,0]};{pos[i,j,1]},'
#         pos_outfile.write(str+'\n')
