from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as cs
import ast2000tools.utils as ut
system = SolarSystem(59529)

print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'
      .format(system.star_mass, system.star_radius))

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
