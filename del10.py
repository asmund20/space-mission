import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.star_population import StarPopulation
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cst

seed = 59529
system = SolarSystem(seed)
stars = StarPopulation(seed=59529)

ss_color = system.star_color        # (r,g,b)
ss_mass = system.star_mass          # [M_sun]
ss_radius = system.star_radius      # [km]
ss_temp = system.star_temperature   # [K]

ss_lum = 4*np.pi*cst.sigma*((ss_radius*1e3)**2) * (ss_temp**4)/cst.L_sun
t_life = 1e10*(1/ss_mass)**3
print(f'Life expectancy: {t_life:.3g} yr')

T = stars.temperatures # [K]
L = stars.luminosities # [L_sun]
r = stars.radii        # [R_sun]

c = stars.colors
s = np.maximum(1e3*(r - r.min())/(r.max() - r.min()), 1.0) # Make point areas proportional to star radii

fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='k', linewidth=0.05)
ax.scatter(ss_temp, ss_lum, c='red', s=20)
plt.figtext(0.4, 0.47, 'Stellaris Skarsgård', fontsize=10)

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

print("SS luminosity", ss_lum, "L_sun")
print(f"SS radius {ss_radius:.3e} km")
print(f"SS mass {ss_mass} M_sun")
print(f"SS surface temp {ss_temp} K")
density = ss_mass*cst.m_sun/(4/3*np.pi*ss_radius**3*1e9)
print(f"SS density {ss_mass*cst.m_sun/(4/3*np.pi*ss_radius**3*1e9)} kg/m³")

T_c = ss_temp + 2*np.pi/3*cst.G*density*1.75*cst.m_p/cst.k_B*ss_radius**2*1e6

print(f"Core temperature {T_c:.3g} K")

plt.show()
