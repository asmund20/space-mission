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
    return doppler_vel(np.asarray(mission.star_doppler_shifts_at_sun))

def vel_spacecraft(dlambda1, dlambda2):
    return doppler_vel(np.asarray([dlambda1, dlambda2]))

def rel_vel_spacecraft_xy(dlambda1, dlambda2):
    phi = np.asarray(mission.star_direction_angles)
    phi = phi/180*np.pi
    # denne matrisen er fæil
    M = 1/np.sin(phi[1]-phi[0])*np.asarray([
        [np.sin(phi[1]), -np.sin(phi[0])],
        [-np.cos(phi[1]), np.cos(phi[0])]])

    """
    denne matrisen er riktig
    M = np.asarray([
        [np.cos(phi[0]), np.cos(phi[1])],
        [np.sin(phi[0]), np.sin(phi[1])],
        ])
    """

    return np.matmul(M, vel_spacecraft(dlambda1, dlambda2)-vel_sun().T)

# d has the shape (d_0, d_1, ..., d_n, d_star)
def triliteration(t: float, d):
    # shape (2, n_planets, n_steps)
    planet_positions = np.load("positions.npy")
    i = int(t/1e-4)

    circ = 2*np.pi*d[0]
    dbue = 1e-4
    min_N = 1000
    N = max(min_N, int(circ/dbue))
    phi = np.linspace(0, 2*np.pi, N, endpoint=False)

    # a circle around Zeron with radius equal to the distance to Zeron
    S = np.asarray([d[0]*np.cos(phi) + planet_positions[0,0,i], d[0]*np.sin(phi) + planet_positions[1,0,i]])

    # the sum of the square errors for the distance to the rest of the planets
    # for each of the points on the circle S
    e = np.zeros(N)

    for j, dj in enumerate(d[1:]):
        j += 1
        A = S.copy()
        A[0,:] = S[0,:]-planet_positions[0,j,i]
        A[1,:] = S[1,:]-planet_positions[1,j,i]

        for k, _ in enumerate(A[0]):
            e[k] += (np.linalg.norm(A[:,k])-dj)**2

    #returning the minimum square error
    i = np.argmin(e)
    return S[:,i]


def main(dt, z, fuel, fuel_consumption, thrust, launch_time=0):
    # finner størrelser for å teste triliteration og rel_vel_spacecraft_xy
    r, v, _, _ = sim_launch(launch_time)

    # gjør greier i mission som er nødvendige for å kunne få distnces og doppler-skifer
    mission.set_launch_parameters(thrust, fuel_consumption, fuel[0], dt*(len(z)-1), r[0], launch_time)
    mission.launch_rocket()
    mission.verify_launch_result(r[-1])

    # avstanden til alle planetene
    d = mission.measure_distances()[:-1]
    print(f"Distances to the planets: {d}")

    dlambda1, dlambda2 = mission.measure_star_doppler_shifts()
    print(f"Measured Doppler-shifts: {dlambda1, dlambda2} nm")
    print(f"Velocity from doppler-effect {rel_vel_spacecraft_xy(dlambda1, dlambda2)} AU")
    print("Actual velocity",v, "AU/yr")
    print(f"Relative differnce: {np.linalg.norm(v-rel_vel_spacecraft_xy(dlambda1, dlambda2))/np.linalg.norm(v)}")

    print(f"Time from launch to completion: {dt*(len(z)-1)} s eller {ut.s_to_yr(dt*(len(z)-1))} yr")
    print(f"Triliteration position: {triliteration(launch_time + ut.s_to_yr(dt*(len(z)-1)), d)} AU")
    print("Actual position", r[-1], "AU")
    print(f"Relative difference: {np.linalg.norm(r[-1]-triliteration(launch_time + ut.s_to_yr(dt*(len(z)-1)), d))/np.linalg.norm(r[-1])}")

def test():
    t = 0
    planet_pos = np.load("positions.npy")[:,:,0]
    n_planets = len(planet_pos[0])
    d = np.zeros(n_planets)

    for i, _ in enumerate(d):
        d[i] = np.linalg.norm(planet_pos[:,i])

    print(d)
    print("Actual posision is (0, 0)")
    print(f"Position from triliteration is {triliteration(t, d)}")
    


def test_for_launch_at():
    try:
        print(f"Launch time: {sys.argv[1]} yr")
        # simulerer launch sett fra planeten
        dt, z, _, _, _, fuel, _, fuel_consumption, thrust = launch()

        main(dt, z, fuel, fuel_consumption, thrust, launch_time=float(sys.argv[1]))
        print("\n")

    except IndexError:
        print("Må kjøres med launch-tidspunkt i år som argument for å få testet ved launch også, f. eks 'python del4.py 0'")
if __name__ == "__main__":
    test_for_launch_at()
    test()
