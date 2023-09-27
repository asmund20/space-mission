from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission


def doppler_vel(dlamba):
    return cs.c*dlamba/656.3e-9

def vel_sun(m):
    return doppler_vel(np.asarray(m.star_doppler_shifts_at_sun)*1e-9)

def main():
    seed = 59529
    system = SolarSystem(seed)
    mission = SpaceMission(seed)

    print(mission.star_direction_angles)
    print(vel_sun(mission))


main()
