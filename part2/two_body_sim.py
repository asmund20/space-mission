import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs

def calculate_orbits():
    seed = 59529
    system = SolarSystem(seed)

    # have checked which planet has the most signifficant pull
    # on the star, that is planet 6.

    # in the reference frame where the star is stationary
    planet_pos_x = system.initial_positions[0,6]
    planet_pos_y = system.initial_positions[1,6]
    planet_pos = np.asarray([planet_pos_x, planet_pos_y])

    planet_vel_x = system.initial_velocities[0,6]
    planet_vel_y = system.initial_velocities[1,6]
    planet_vel = np.asarray([planet_vel_x, planet_vel_y])

    planet_mass = system.masses[6]
    star_mass = system.star_mass

    # the star is in the origin, and therefore does not need to be included
    # in the sum for the vectors when calculating the CM
    CM = 1/(planet_mass+star_mass)*planet_mass*planet_pos

    print(CM)
    print(planet_pos)

    # will now change from the star-reference-system to the center of mass reference-frame.
    # the new positions for the star and the planet will now be new_pos = old_pos - CM

    planet_pos -= CM
    str_pos = -CM

    # finding the velicities in the new reference-frame


if __name__ == "__main__":
    calculate_orbits()
