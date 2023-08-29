from ast2000tools.solar_system import SolarSystem
system = SolarSystem(59529)

print('My system has a {:g} solar mass star with a radius of {:g} kilometers.'
      .format(system.star_mass, system.star_radius))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU with mass {} solar masses'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx], system.masses[planet_idx]))
