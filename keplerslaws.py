#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cs


seed = 59529
system = SolarSystem(seed)

num_planets = system.number_of_planets

a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
M_s = system.star_mass              # Massen til Stellaris Skarsgard
M_zeron = system.masses[0]          # Massen til Zeron

# Perioden til Zeron: bruker denne til a finne tiden det tar for 30 omlop rundt Stel. Skars.
period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(cs.G_sol*(M_s+M_zeron)))
dt = 1e-4
t = np.arange(0, 30*period_zeron,dt)

def periods(pos):
    periods = np.zeros(num_planets)
    for i in range(num_planets):
        x = pos[0,i,:]
        r = np.sqrt(pos[0,i,:]**2 + pos[1,i,:]**2)
        index = np.argwhere(abs(x/r - np.cos(system.initial_orbital_angles[i])) <= 1e-5)
    
        j = np.argwhere(index[1:,0]-index[:-1,0] != 1)
        periods[i] = np.mean(index[j[0,0]:j[1,0]])*dt
    return periods


pos = np.load('positions.npy')

dA = np.zeros(N)
for i in range(N-1):
    dA[i] = np.linalg.norm(np.cross(pos[:,0,i], pos[:,0,i+1]))/2

n = 10  # Antall arealelement som skal inkluderes
# Areal utspendt ved perihel
dA_peri = np.sum(dA[int(0.5*period_zeron/dt):int(0.5*period_zeron/dt)+n+1])
# Areal utspendt ved aphel
dA_ap = np.sum(dA[0:n+1])

# Avstand dekket i løpet av n*dt ved perihel
dist_peri = np.sum([np.linalg.norm(pos[:,0,i+1]-pos[:,0,i]) \
                     for i in range(int(0.5*period_zeron/dt), int(0.5*period_zeron/dt)+n+1)])

# Avstand dekket i løpet av n*dt ved aphel
dist_ap = np.sum([np.linalg.norm(pos[:,0,i+1]-pos[:,0,i]) \
                  for i in range(0,n+1)])

vel_peri = dist_peri/(n*dt)     # Gjennomsnittsfart ved perihel
vel_ap = dist_ap/(n*dt)         # Gjennomsnitssfart ved aphel

# Regner perioden til planetene numerisk og ved Keplers 3.lov/Newton
num_periods = periods(pos)
kepler_periods = [np.sqrt((4*np.pi**2 * system.semi_major_axes[i]**3)/(cs.G_sol*(M_s))) for i in range(num_planets)]
newton_periods = [np.sqrt((4*np.pi**2 * system.semi_major_axes[i]**3)/(cs.G_sol*(M_s + system.masses[i]))) for i in range(num_planets)]


print(
f"""
#=====================#
#   KEPLERS 2. LOV    #
#=====================#

Med tidsintervall dt = {n*dt} yr, utspennes et areal ved
========================================================
Perihel: dA_p = {dA_peri} AU^2
Aphel:   dA_a = {dA_ap} AU^2
--------------------------------------------------------
Diff:    dA_p - dA_a = {(dA_peri-dA_ap):g} AU^2


I løpet av tiden dt, dekket planeten en avstand ved
========================================================
Perihel: dist_p = {dist_peri} AU
Aphel:   dist_a = {dist_ap} AU
--------------------------------------------------------
Diff:    dist_p - dist_a = {(dist_peri-dist_ap):g} AU


Gjennomsnittsfart
========================================================
Perihel: v_p = {vel_peri} AU/yr
Aphel:   v_a = {vel_ap} AU/yr
--------------------------------------------------------
Diff:    v_p - v_a = {(vel_peri-vel_ap):g} AU/yr



#=====================#
#   KEPLERS 3. LOV    #
#=====================#

Periodene til planetene:
-----------------------------------------------------------------
        KEPLER       |       NEWTON        |       NUMERISK      
"""
)
for kep,new,num in zip(kepler_periods, newton_periods, num_periods):
    print(f'{kep:^17.8g} yr |{new:^17.8g} yr |{num:^17.8g} yr ')

