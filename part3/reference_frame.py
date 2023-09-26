#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

from matplotlib import pyplot as plt, patches
import numpy as np
import scipy.constants as cs
from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as utils
import ast2000tools.constants as astconst

seed = 59529

system = SolarSystem(seed)

# the time for the start of the launch
launch_time = 0
# the angle relative to the x-axis determining the launch position on the planet
launch_angle = 0

r1_norm = np.load('rocket_position.npy')

planet_positions = np.genfromtxt('positions.txt', delimiter=",")

# Rotasjonshastighet til Zeron
omega = 2*np.pi/(system.rotational_periods[0]*60*60*24)

# må endres !!!!!!
# Posisjonen til Zeron
r0 = np.zeros((len(r1_norm),2))
r0[0] = system.initial_positions[0,0], system.initial_positions[1,0]
r0[0] = utils.AU_to_m(r0[0])

# Hastigheten til Zeron
v0 = np.zeros((len(r1_norm),2))
v0[0] = system.initial_velocities[0,0], system.initial_velocities[1,0]
v0[0] = utils.AU_pr_yr_to_m_pr_s(v0[0])
# til hit!!!!

# Posisjonen til raketten relativt til Zeron
r1 = np.zeros((len(r1_norm),2))
r1[0] = r1_norm[0], 0

# Tid
t = 0
dt = 1e-3 # s

# Massen til stjerna Stellaris Skarsgaard
M = system.star_mass*astconst.m_sun


for i in range(len(r1_norm)-1):

    # Regner r1 komponenten til posisjonen
    # legger til ønsket vinkel definert fra x-aksen til launch-posisjonen
    r1[i+1] = np.array([r1_norm[i+1]*np.cos(omega*t), r1_norm[i+1]*np.sin(omega*t)])

    # Numerisk integrasjon
    # Akselerasjonen til Zeron
    a = - cs.G*M*r0[i]/np.linalg.norm(r0[i])**3
    v0[i+1] = v0[i]+a*dt
    r0[i+1] = r0[i]+v0[i+1]*dt # Euler-Cromer

    t += dt

# Konvertere til AU
r = utils.m_to_AU(r0+r1)
r0 = utils.m_to_AU(r0)

# Slutthastighet i AU/yr
vf = (r[-1]-r[-2])/utils.s_to_yr(dt)

print(f'Final velocity rocket: {vf} AU/yr')
print(f'Final position rocket: {r[-1]} AU')

"""
# Plotte banen til Zeron og raketten
fig = plt.figure()
fig.suptitle('Banene fulgt av raketten og Zeron i det inertielle referansesytemet')
ax = fig.add_subplot()
zeron = patches.Circle((r0[-1,0], r0[-1,1]), radius=r[0,0]-r0[0,0], color='green', label='Zeron')
ax.add_patch(zeron)

l = 1.7*r0[-1,1]
ax.set_ylim(-l, l)
ax.set_xlim(r0[0,0]-l, r0[0,0] + l)
ax.set_xlabel('x [AU]')
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticklabels([str(round(float(label), 6)) for label in labels], rotation=-45)
ax.set_ylabel('y [AU]')

# ax.plot(r0[-1,0], r0[-1,1], 'o', color='orange', markersize=375, label='Zeron')
ax.plot(r0[:,0], r0[:,1], 'k:', label='Banen til Zeron')
ax.plot(r[:,0], r[:,1], 'k', label='Banen til raketten')
ax.quiver(r[-1,0], r[-1,1], vf[0],vf[1], color='red', scale=50, label='Hastighetsvektoren etter fullført oppskytning')

plt.legend()
plt.tight_layout()

plt.show()
"""
