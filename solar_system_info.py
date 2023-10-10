from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as cs
import ast2000tools.utils as ut
import numpy as np
mission = SpaceMission(59529)
system = SolarSystem(59529)

print('My system has a {:g} solar mass star with a radius of {:g} kilometers and temperature of {:g} K.'
      .format(system.star_mass, system.star_radius, system.star_temperature))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU with mass {} solar masse and radius {} km'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx], system.masses[planet_idx], system.radii[planet_idx]))

    print(f"Planet {planet_idx} initial_angle: {system.initial_orbital_angles[planet_idx]:.2f}, aphelion angle: {system.aphelion_angles[planet_idx]:.2f}")

print("info om solsystemet:")
print(f"""
Startposisjon {system.initial_positions[0, 0]},{system.initial_positions[1, 0]}
eksentrisitet: {system.eccentricities[0]}
phi: {system.initial_orbital_angles[0]}
banevinkel: {system.aphelion_angles[0]}
starthastighet: {system.initial_velocities[0, 0],system.initial_velocities[1, 0]}
døgn: {system.rotational_periods[0]}""")


g = system.masses*cs.m_sun /ut.AU_to_m(system.semi_major_axes)**2
for i, gi in enumerate(g):
    print(f"planet {i} bidrar til gravitasjonsfeltet i sentrum til Stellaris Skarsgård med {gi} ganger en konstant")

print(f'Star angles used for Doppler: phi1 = {mission.star_direction_angles[0]:.6g}, phi2 = {mission.star_direction_angles[1]:.6g}')
l = system.semi_major_axes[1]*np.sqrt(system.masses[1]/(10*system.star_mass))
print(f'l = {ut.AU_to_km(l)}')
print(f'L = {system.radii[1]*640*180/70/np.pi} = {ut.km_to_AU(system.radii[1]*640*180/70/np.pi)}')
print(f'Tvekne eksentrisitet: {system.eccentricities[1]}')




print(f"Banefart Zeron: {np.linalg.norm(system.initial_velocities[:,0])} AU/yr")
print(f"Banefart Tvekne: {np.linalg.norm(system.initial_velocities[:,1])} AU/yr")
