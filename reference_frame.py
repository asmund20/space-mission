import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as utils
import ast2000tools.constants as astconst

seed = 59529

system = SolarSystem(seed)

r1_norm = []
# Leser inn avstanden fra Zeron til rakett fra vellykket launch
with open('rocket_position.txt', 'r') as infile:
    lines = infile.readlines()
    for line in lines:
        r1_norm.append(float(line.strip('\n')))
r1_norm = np.array(r1_norm)

# Rotasjonshastighet til Zeron
omega = 2*np.pi/(system.rotational_periods[0]*60*60*24)

# Posisjonen til Zeron
r0 = np.zeros((len(r1_norm),2))
r0[0] = system.initial_positions[0,0], system.initial_positions[1,0]
r0[0] = utils.AU_to_m(r0[0])

# Hastigheten til Zeron
v0 = np.zeros((len(r1_norm),2))
v0[0] = system.initial_velocities[0,0], system.initial_velocities[1,0]
v0[0] = utils.AU_pr_yr_to_m_pr_s(v0[0])

# Posisjonen til raketten relativt til Zeron
r1 = np.zeros((len(r1_norm),2))
r1[0] = r1_norm[0], 0

# Tid
t = 0
dt = 1e-3 # s

# Massen til stjerna
M = system.star_mass*astconst.m_sun


for i in range(len(r1_norm)-1):

    r1[i+1] = np.array([r1_norm[i+1]*np.cos(omega*t), r1_norm[i+1]*np.sin(omega*t)])

    # Akselerasjonen til Zeron
    a = - cs.G*M*r0[i]/np.linalg.norm(r0[i])**3
    v0[i+1] = v0[i]+a*dt
    r0[i+1] = r0[i]+v0[i+1]*dt

    t += dt

# Konvertere til AU
r = utils.m_to_AU(r0+r1)
r0 = utils.m_to_AU(r0)

print(f'Final velocity rocket: {(r[-1]-r[-2])/utils.s_to_yr(dt)} AU/yr')
print(f'Final position rocket: {r[-1]} AU')

# Plotte sola, banen til Zeron og raketten
plt.plot(0,0,'yo', markersize=10)
plt.axis('equal')
plt.plot(r0[:,0], r0[:,1])
plt.plot(r[:,0], r[:,1])
plt.show()
