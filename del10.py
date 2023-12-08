import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.star_population import StarPopulation
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cst
import scipy.constants as cs

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

plt.style.use("dark_background")
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

X_H = 0.745
X_CNO = 0.002

eps_0pp = 1.08e-12
eps_0CNO = 8.24e-31

e_pp = eps_0pp * (X_H**2) * density * (T_c/1e6)**4
e_CNO = eps_0CNO * X_H * X_CNO * density * (T_c/1e6)**20

e = e_pp + e_CNO
print(f'Reaction-rate core: {e:.3g} W/kg')

core_mass = density * 4*np.pi*(0.2*ss_radius*1e3)**3
print(f'Core mass: {core_mass:.3g} kg')

lum_core = e*core_mass
print(f'Luminosity estimate from reactions in core: {lum_core:.3g} W')
print(f'Luminosity in L_sun: {lum_core/cst.L_sun:.3g} L_sun')
print(f'T_eff: {(lum_core/4/np.pi/(ss_radius*1e3)**2/cst.sigma)**(1/4):.3g} K')
print(cst.G*0.44*cst.m_sun/(1882*1e3)**2)

m_wd = 1.4*ss_mass/8
R_wd = (3/2/np.pi)**(4/3)*cs.h**2/20/cs.m_e/cst.G*(1/2/cs.m_p)**(5/3)*(m_wd*cst.m_sun)**(-1/3)
density_wd = m_wd*cst.m_sun/(4/3*np.pi*R_wd**3)

print(f"Mass WD {m_wd} m_sun")
print(f"Radius WD {R_wd*1e-3} km")
print(f"Density WD {density_wd:.4e} kg/m³")
print(f"En liter med hvit dverg veier {density_wd*1e-3:.4e} kg")

plt.show()
