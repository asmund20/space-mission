from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

def doppler_vel(dlamba):
    return cs.c*dlamba/656.3e-9

def vel_sun():
    return doppler_vel(np.asarray(mission.star_doppler_shifts_at_sun)*1e-9)

def vel_planet(dlambda1, dlambda2):
    return doppler_vel(np.asarray([dlambda1, dlambda2]))

def rel_vel_planet(dlambda1, dlambda2):
    return vel_planet(dlambda1, dlambda2) - vel_sun()

def rel_vel_planet_xy(dlambda1, dlambda2):
    phi = mission.star_direction_angles
    M = 1/np.sin(phi[1]-phi[0])*np.asarray([
        [np.sin(phi[1]), -np.sin(phi[0])],
        [-np.cos(phi[1]), np.cos(phi[0])]])

    return np.matmul(M, rel_vel_planet(dlambda1, dlambda2))

# d has the shape (d_0, d_1, ..., d_n, d_star)
def triliteration(t: float, d):
    # shape (2, n_planets, n_steps)
    planet_positions = np.load("position.npy")
    i = int(t/1e-4)

    phi = np.linspace(0, 2*np.pi, 1000, endpoint=False)

    # a sircle around Zeron with radiu equal to the distance to Zeron
    S = np.asarray([d[0]*np.cos(phi),d[0]*np.sin(phi)]) + d[:,0,i]

    # the sum of the square errors for the distance to the two other planets
    e = (S-d[:,1,i])**2+(S-d[:,2,i])**2

    i = np.argmin(e)[0]
    return S[i]


def main():

    print(mission.star_direction_angles)
    print(vel_sun())
    print(rel_vel_planet(0,0))
    print(rel_vel_planet_xy(0, 0))
    l1, l2 = mission.star_doppler_shifts_at_sun
    l1 = l1 *1e-9
    l2 = l2 *1e-9
    print(rel_vel_planet_xy(l1, l2))


main()
