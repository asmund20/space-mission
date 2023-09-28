from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from reference_frame import sim_launch
from simulate_launch import launch
import sys

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

def doppler_vel(dlamba):
    return -cs.c_AU_pr_yr*dlamba/656.3

def vel_sun():
    #phi = mission.star_direction_angles
    #M = 1/np.sin(phi[1]-phi[0])*np.asarray([
        #[np.sin(phi[1]), -np.sin(phi[0])],
        #[-np.cos(phi[1]), np.cos(phi[0])]])
    #return np.matmul(M, np.asarray(mission.star_doppler_shifts_at_sun))
    return doppler_vel(np.asarray(mission.star_doppler_shifts_at_sun))

def vel_spacecraft(dlambda1, dlambda2):
    #phi = mission.star_direction_angles
    #M = 1/np.sin(phi[1]-phi[0])*np.asarray([
        #[np.sin(phi[1]), -np.sin(phi[0])],
        #[-np.cos(phi[1]), np.cos(phi[0])]])
    #return np.matmul(M, doppler_vel(np.asarray([dlambda1, dlambda2])))
    return doppler_vel(np.asarray([dlambda1, dlambda2]))

def rel_vel_spacecraft_xy(dlambda1, dlambda2):
    phi = mission.star_direction_angles
    M = 1/np.sin(phi[1]-phi[0])*np.asarray([
        [np.sin(phi[1]), -np.sin(phi[0])],
        [-np.cos(phi[1]), np.cos(phi[0])]])

    return np.matmul(M, vel_spacecraft(dlambda1, dlambda2)-vel_sun())
    #return vel_planet(dlambda1, dlambda2)-vel_sun()

# d has the shape (d_0, d_1, ..., d_n, d_star)
def triliteration(t: float, d):
    # shape (2, n_planets, n_steps)
    planet_positions = np.load("positions.npy")
    i = int(t/1e-4)

    phi = np.linspace(0, 2*np.pi, 1000, endpoint=False)

    # a circle around Zeron with radius equal to the distance to Zeron
    S = np.asarray([d[0]*np.cos(phi) + planet_positions[0,0,i], d[0]*np.sin(phi) + planet_positions[1,0,i]])
    #S[0] += planet_positions[0,0,i]
    #S[1] += planet_positions[1,0,i]

    # the sum of the square errors for the distance to the rest of the planets
    # for each of the points on the circle S
    e = np.zeros(len(S[0]))
    for j, dj in enumerate(d[1:]):
        A = S.copy()
        A[0,:] = S[0,:]-planet_positions[0,j,i]
        A[1,:] = S[1,:]-planet_positions[1,j,i]

        for k, a in enumerate(A):
            e[k] -= (np.linalg.norm(a)-dj)**2

    i = np.argmin(e)
    return S[:,i]


def main(dt, z, fuel, fuel_consumption, thrust, launch_time=0):
    r, v, r0 = sim_launch(launch_time)

    mission.set_launch_parameters(thrust, fuel_consumption, fuel[0], dt*(len(z)-1), r[0], launch_time)
    mission.launch_rocket()
    mission.verify_launch_result(r[-1])

    dlambda1, dlambda2 = mission.measure_star_doppler_shifts()
    print(f"Velocity from doppler-effect {rel_vel_spacecraft_xy(dlambda1, dlambda2)}")
    print("Actual velocity",v)

    d = mission.measure_distances()
    print(f"Time from launch to completion: {dt*(len(z)-1)} s eller {ut.s_to_yr(dt*(len(z)-1))} yr")
    print(f"Triliteration position: {triliteration(launch_time + ut.s_to_yr(dt*(len(z)-1)), d)}")
    print("Actual position", r[-1])

if __name__ == "__main__":
    print("Launch time:", sys.argv[1])
    dt, z, vz, az, mass, fuel, esc_vel, fuel_consumption, thrust = launch()
    launch_times = np.arange(0,10,1)

    main(dt, z, fuel, fuel_consumption, thrust, launch_time=float(sys.argv[1]))

    print("\n")
