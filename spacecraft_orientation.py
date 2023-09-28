from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from reference_frame import sim_launch
from simulate_launch import launch

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

def doppler_vel(dlamba):
    return -cs.c_AU_pr_yr*dlamba/656.3

def vel_sun():
    return doppler_vel(np.asarray(mission.star_doppler_shifts_at_sun))

def vel_planet(dlambda1, dlambda2):
    return doppler_vel(np.asarray([dlambda1, dlambda2]))

def rel_vel_planet(dlambda1, dlambda2):
    return vel_planet(dlambda1, dlambda2)-vel_sun()

def rel_vel_planet_xy(dlambda1, dlambda2):
    phi = mission.star_direction_angles
    M = 1/np.sin(phi[1]-phi[0])*np.asarray([
        [np.sin(phi[1]), -np.sin(phi[0])],
        [-np.cos(phi[1]), np.cos(phi[0])]])

    return np.matmul(M, rel_vel_planet(dlambda1, dlambda2))

# d has the shape (d_0, d_1, ..., d_n, d_star)
def triliteration(t: float, d):
    # shape (2, n_planets, n_steps)
    planet_positions = np.load("positions.npy")
    i = int(t/1e-4)

    phi = np.linspace(0, 2*np.pi, 1000, endpoint=False)

    # a circle around Zeron with radiu equal to the distance to Zeron
    S = np.asarray([d[0]*np.cos(phi),d[0]*np.sin(phi)])
    S[0] += planet_positions[0,0,i]
    S[1] += planet_positions[1,0,i]

    # the sum of the square errors for the distance to the rest of the planets
    e = S.copy()
    for j, dj in enumerate(d[1:]):
        A = S.copy()
        A[0,:] = S[0,:]-planet_positions[0,j,i]
        A[1,:] = S[1,:]-planet_positions[1,j,i]

        for k, a in enumerate(A):
            A[k] = np.linalg.norm(a)

        A -= dj

        e += A**2

    i = np.argmin(e)
    return S[:,i]


def main():

    print(mission.star_direction_angles)
    print(vel_sun())
    print(rel_vel_planet(0,0))
    print(rel_vel_planet_xy(0, 0))
    l1, l2 = mission.star_doppler_shifts_at_sun
    l1 = l1 *1e-9
    l2 = l2 *1e-9
    print(rel_vel_planet_xy(l1, l2))

    launch_time = 3
    dt, z, vz, az, mass, fuel, esc_vel, fuel_consumption, thrust = launch()
    r, v, r0 = sim_launch(launch_time)

    mission.set_launch_parameters(thrust, fuel_consumption, fuel[0], dt*len(mass), r[0], launch_time)
    mission.launch_rocket()
    mission.verify_launch_result(r[-1])
    dlambda1, dlambda2 = mission.measure_star_doppler_shifts()
    d = mission.measure_distances()
    print(rel_vel_planet(dlambda1, dlambda2))
    print(v)
    print(triliteration(launch_time + dt*len(mass)/(60**2*365), d))
    print(r[-1])

main()
