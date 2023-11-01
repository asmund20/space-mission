from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
#import matplotlib.pyplot as plt
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from reference_frame import sim_launch
from del5 import plan_trajectory
import numpy as np
import copy


seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)


def initiate_circular_orbit():

    ### SHORTCUT ###
    code_unstable_orbit = 67311
    code_orientation_data = 9851
    shortcut = SpaceMissionShortcuts(mission, [code_orientation_data])
    unstable_orbit = SpaceMissionShortcuts(mission, [code_unstable_orbit])
    ################

    # Read planet positions and velocitites from part 2
    pos_planets = np.load('positions.npy')
    vel_planets = np.load("velocities.npy")

    # Read rocket altitude from launch in part 1
    rocket_altitude = np.load('rocket_position.npy')
    launch_duration = ut.s_to_yr(1e-3*(len(rocket_altitude)))

    star_mass = system.star_mass
    planet_masses = system.masses

    timesteps_planets = len(pos_planets[0,0])
    known_times = np.linspace(0, timesteps_planets*1e-4, timesteps_planets)

    time_start_launch, phi0, travel_duration, endpoint = plan_trajectory()
    rocket_positions_during_launch, rocket_velocity_after_launch, _, _ = sim_launch(time_start_launch, phi0)
    fuel_consumption, thrust, fuel = np.load('rocket_specs.npy')

    # Verify launch
    mission.set_launch_parameters(thrust, fuel_consumption, fuel, ut.yr_to_s(launch_duration), rocket_positions_during_launch[0], time_start_launch)
    mission.launch_rocket()
    mission.verify_launch_result(rocket_positions_during_launch[-1])

    ### SHORTCUT ###
    sc_position, sc_velocity, sc_motion_angle = shortcut.get_orientation_data()
    mission.verify_manual_orientation(sc_position, sc_velocity, sc_motion_angle)

    time = 25.4123
    planet_idx = 1

    unstable_orbit.place_spacecraft_in_unstable_orbit(time, planet_idx)
    land = mission.begin_landing_sequence()
    ################

    # Free fall some time before injection maneuver
    time_step = 1000
    n = 1000
    positions = []
    for _ in range(35):
        land.fall(time_step)
        pos = land.orient()[1]
        positions.append(pos)
    
    # Calculates injection maneuver boost
    t0, r0, v0 = land.orient()
    theta = np.angle(complex(r0[0], r0[1]))
    v_stable = np.sqrt(cs.G*cs.m_sun*planet_masses[1]/np.linalg.norm(r0))
    e_theta = np.array([-r0[1]/np.linalg.norm(r0), r0[0]/np.linalg.norm(r0), 0])
    dv_inj = e_theta*v_stable - v0

    print('#'*20)
    print(f'v0 = {v0}')
    print(f'v_stable*e_theta = {e_theta*v_stable}')
    print(f'theta = {theta}')
    print(f'dv_inj = {dv_inj}')
    print('#'*20)

    land.boost(dv_inj)

    return land, pos_planets


# calculates the spherical coordinate for a point on Tvekne's surface.
# takes the current position (theta, phi) and the time after which to return the position.
# returns: theta, phi
def landing_site_position(theta, phi, time):
    omega = 1/system.rotational_periods[1]/60**2/24
    return theta, phi+time*omega


# this changes the position of your spacecraft, call with a copy if 
# you want to avoid that
def avg_dist(landing_sequence, runtime):
    l = [np.linalg.norm(landing_sequence.orient()[1])]

    dt = max(1e3, runtime/1000)
    t = 0

    while t < runtime:
        landing_sequence.fall(dt)
        _, p, _ = landing_sequence.orient()
        l.append(np.linalg.norm(p))
        t += dt

    return sum(l)/len(l)


def verify_not_in_atmosphere(landing_sequence):
    time_per_sim = 1e6
    tolerance = np.linalg.norm(landing_sequence.orient()[1])/10

    landing_sequence = copy.deepcopy(landing_sequence)

    return avg_dist(landing_sequence, time_per_sim) - avg_dist(landing_sequence, time_per_sim) < tolerance


def main():
    landing_sequence, planet_positions = initiate_circular_orbit()
    t, p, v = landing_sequence.orient()
    print(t, p, v)
    landing_sequence.fall(100)
    t, p, v = landing_sequence.orient()
    print(t, p, v)
    landing_sequence.boost(-0.726*v)

    print(avg_dist(landing_sequence, 100))
    print(verify_not_in_atmosphere(landing_sequence))

    landing_sequence.fall(1e6)

    N = 10
    orbit_for = 700000
    dt = orbit_for/N

    for i in range(N):
        landing_sequence.look_in_direction_of_planet(1)
        landing_sequence.take_picture(filename=f"landing_picture{i}.xml")
        landing_sequence.fall(dt)



if __name__=="__main__":
    main()
    ...
