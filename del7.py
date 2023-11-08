########################
#  IKKE BRUKT KODEMAL  #
########################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import ast2000tools.constants as cs 
import ast2000tools.utils as ut
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from numba import jit
import tqdm
import copy
import sys

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)
np.random.seed(seed)

vmax_c = 1e4/cs.c

# The chemical formula of the gasses assosiated with the
# different lmbda0 values.
spectral_lines = {
    632: 'O2', 690: 'O2', 760: 'O2', 720: 'H2O',
    820: 'H2O', 940: 'H2O', 1400: 'CO2', 1600: 'CO2',
    1660: 'CH4', 2200: 'CH4', 2340: 'CO', 2870: 'N2O'
}

# Information about the gasses.
gasses = {
    'O2': {'name': 'Oxygen', 'A': 32}, 'H2O': {'name': 'Water vapor', 'A': 18},
    'CO2': {'name': 'Carbon dioxide', 'A': 44}, 'CH4': {'name': 'Methane', 'A': 16},
    'CO': {'name': 'Carbon monoxide', 'A': 28}, 'N2O': {'name': 'Nitrous dioxide', 'A': 44}
}


def temperature(r):
    """Temperature as a function of distance from center r"""
    # Mean molecular mass
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    # Temperature on surface
    T0 = 271
    # Density on surface
    rho0 = system.atmospheric_densities[1]
    # Radius of Tvekne
    r0 = system.radii[1]*1e3
    # Mass of Tvekne
    M_T = system.masses[1]*cs.m_sun
    # Gamma vlue for adiabatic gas
    gamma = 1.4

    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    if r > r_iso:
        T = T0/2
    else:
        T = T0 - (gamma-1)/gamma * mu*cs.G*M_T/cs.k_B * (1/r0 - 1/r)
    
    return T

def pressure(r):
    """Pressure as a function of distance from center r"""
    # Mean molecular mass
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    # Temperature on surface
    T0 = 271
    # Density on surface
    rho0 = system.atmospheric_densities[1]
    # Radius of Tvekne
    r0 = system.radii[1]*1e3
    # Mass of Tvekne
    M_T = system.masses[1]*cs.m_sun
    # Gamma vlue for adiabatic gas
    gamma = 1.4
    # Pressure at surface level
    p0 = rho0*cs.k_B*T0/mu
    
    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    if r > r_iso:
        p_iso = p0 * (T0/temperature(r_iso))**(gamma/(1-gamma))
        p = p_iso * np.exp(-(2*mu*cs.G*M_T)/cs.k_B/T0 * (1/r_iso - 1/r))
    else:
        p = p0 * (T0/temperature(r))**(gamma/(1-gamma))
    
    return p

def density(r):
    """Density as a function of distance from center r"""
    # Mean molecular mass
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    return pressure(r)*mu/cs.k_B/temperature(r)
    
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
    
    # Verify launch: here we use the parameters as before just printed out in
    # the terminal and pasted in here to save time and avoid needing to upload the same
    # .py-files as before.
    mission.set_launch_parameters(5109888.470669283, 946.9636, 299999.99999834, 307.6, [-3.52242556, -0.06816988], 23.195500000000003)
    mission.launch_rocket()
    mission.verify_launch_result([-3.52243009, -0.06822316])

    ### SHORTCUT ###
    sc_position, sc_velocity, sc_motion_angle = shortcut.get_orientation_data()
    mission.verify_manual_orientation(sc_position, sc_velocity, sc_motion_angle)

    # opproximately where we would have reached the planet if we managed to do it ourselves
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

    land.boost(dv_inj)

    return land, pos_planets


# calculates the spherical coordinate for a point on Tvekne's surface after a time.
# takes the current position (theta, phi) and the time after which to return the position.
# returns: theta, phi


#landing_site is [r, theta, phi, t]
def landing_site_position(landing_site, t):
    omega = 1/system.rotational_periods[1]/60**2/24
    landing_site[2] = landing_site[2] + (t-landing_site[3])*omega
    landing_site[3] = t


def initiate_orbit():
    landing_sequence, planet_positions = initiate_circular_orbit()
    landing_sequence.fall(100)
    _, _, v = landing_sequence.orient()

    # slowing down so that the spacecraft will get closer to the planet
    landing_sequence.boost(-0.8*v)

    # time-step in which we check if the spacecrft is closer than the desired distance
    dt = 100
    # the distance we want to the surface of the planet
    desired_h = 1e6 #m
    desired_r = system.radii[1]*1e3+desired_h

    while desired_r < np.linalg.norm(landing_sequence.orient()[1]):
        landing_sequence.fall(dt)

    _, p, v = landing_sequence.orient()
    # Calculates injection maneuver boost
    planet_mass = system.masses[1]
    theta = np.angle(complex(p[0], p[1]))
    v_stable = np.sqrt(cs.G*cs.m_sun*planet_mass/np.linalg.norm(p))
    e_theta = np.array([-p[1]/np.linalg.norm(p), p[0]/np.linalg.norm(p), 0])
    dv_inj = e_theta*v_stable - v

    # making sure that te direction of travel stays the same, for relism-purposes
    if np.linalg.norm(dv_inj) > np.linalg.norm(v):
        dv_inj = -e_theta*v_stable-v

    landing_sequence.boost(dv_inj)

    return landing_sequence

# deploy_parachute is the height above the ground to deploy the parachute.
# initiate_at is the time in seconds to wait before the initiation_boost is performed
def simulate_landing(landing_sequence, deploy_parachute, parachute_area, initiate_at, initiation_boost):
    landing_sequence = copy.deepcopy(landing_sequence)
    landing_sequence.fall(initiate_at)

    t, pos, v = landing_sequence.orient()

    v += initiation_boost * v/np.linalg.norm(v)

    area = mission.lander_area
    
    g = lambda r: - cs.G * system.masses[1]*cs.m_sun * r /np.linalg.norm(r)**3

    drag = lambda r, v_drag, rho, A: -1/2 * rho * A * np.linalg.norm(v_drag) * v_drag

    w = lambda r: np.cross(np.array([0, 0, 1/system.rotational_periods[1]/(60**2*24)]), r)

    positions = [pos.copy()]
    velocities = [v.copy()]
    times = [t]
    accelerations = [np.array([0, 0, 0])]
    d = [0]

    dt = 0.1
    deployed_parachute = False
    final_dt = False
    #deployed_parachute = True

    count = 0

    print("starting integration")
    while np.linalg.norm(pos) > system.radii[1]*1e3 and count < 1e6:

        if np.linalg.norm(pos)-system.radii[1]*1e3 <= deploy_parachute and not deployed_parachute:
            area = max(area, parachute_area)
            
            print(f"Deployed parachute at t: {t}, h: {np.linalg.norm(pos)-system.radii[1]*1e3}, v: {np.linalg.norm(v)}")
            deployed_parachute = True
            dt = 1e-4

        if deployed_parachute and not final_dt:
            count += 1

        if count and count > 1e4 and not final_dt:
            dt = 1e-2
            final_dt = True

        current_drag = drag(pos, v-w(pos), density(np.linalg.norm(pos)), area)
        
        if np.linalg.norm(current_drag) > 2.5e5 and deployed_parachute:
            raise RuntimeError(f"Parachute broke due to too high of a force of {np.linalg.norm(current_drag)/2.5e5:.6f} times its capacity")

        a = g(pos) + current_drag/mission.lander_mass
        d.append(np.linalg.norm(drag(pos, v-w(pos), density(np.linalg.norm(pos)), parachute_area)))

        v += a*dt
        pos += v*dt
        t += dt

        positions.append(pos.copy())
        velocities.append(v.copy())
        accelerations.append(a.copy())
        times.append(t)

    return np.array(times), np.array(positions), np.array(velocities), d, np.array(accelerations)

def trial_and_error(landing_sequence, wait_time, boost):
    radius = system.radii[1]*1e3
    desired_landing_spot = np.array([radius, 0, 0.44028 * np.pi, 163850])
    A = 2*cs.G*system.masses[1]*cs.m_sun*mission.lander_mass/density(radius)/radius**2/3**2

    t, position, velocity, drag_force, acceleration = simulate_landing(landing_sequence, 1000, A, wait_time, boost)

    landing_site_position(desired_landing_spot, t[-1])

    final_radial_velocity = np.dot(position[-1], velocity[-1])/np.linalg.norm(position[-1])
    landing_site_angle = np.angle(complex(position[-1,0], position[-1,1]))
    diff_angle = desired_landing_spot[2]-landing_site_angle
    missed_with = radius*diff_angle

    print("Missed desired landing spot with", missed_with/1e3, "km")
    
    print(final_radial_velocity)
    plotting(t, position, velocity, drag_force, acceleration, desired_landing_spot[2])


def plotting(t, position, velocity, drag_force, acceleration, desired_landing_phi):
    radius = system.radii[1]*1e3
    A = 2*cs.G*system.masses[1]*cs.m_sun*mission.lander_mass/density(radius)/radius**2/3**2

    print("parachute_area: ", A)
    print("Final acceleration:", acceleration[-1]) 

    # height-plot
    plt.plot(t, np.linalg.norm(position, axis=1)/1e3-radius/1e3)
    plt.title("Distance from the surface of Tvekne")
    plt.xlabel("time [s]")
    plt.ylabel("height [km]")

    # speed-plot
    plt.figure()
    plt.plot(t, np.linalg.norm(velocity, axis=1))
    plt.title("Lander speed")
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")

    #acceleration-plot
    plt.figure()
    plt.plot(t, np.linalg.norm(acceleration, axis=1))
    plt.title("lander acceleration")
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [m/s²]")

    # position-plot
    plt.figure()
    plt.scatter(radius/1e3*np.cos(desired_landing_phi), radius/1e3*np.sin(desired_landing_phi), label="desired landing-spot", color="green")
    plt.plot(position[:,0]/1e3, position[:,1]/1e3, label="Lander", color="orange")
    angle = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    plt.fill(radius/1e3*np.cos(angle),radius/1e3*np.sin(angle), color="blue", label="Tvekne")
    plt.axis("equal")
    plt.title("The landers trajectory and Tvekne")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.legend()

    plt.show()


def land4real(landing_sequence, falltime, initiation_boost, parachute_area, desired_landing_phi):

    landing_sequence.fall(falltime)
    t, pos, v = landing_sequence.orient()

    # initiation_boost_angle = np.angle(complex(pos[0], pos[1]))
    # dv = initiation_boost * np.array([np.cos(initiation_boost_angle), np.sin(initiation_boost_angle), 0])
    dv = initiation_boost * v/np.linalg.norm(v)
    landing_sequence.launch_lander(dv)
    
    landing_sequence.look_in_direction_of_planet(1)
    landing_sequence.start_video()
    
    landing_sequence.adjust_parachute_area(parachute_area)
    
    dt = 0.1
    h = 1000
    positions = []
    velocities = []
    time = []
    while np.linalg.norm(pos) > system.radii[1]*1e3 + h:
        landing_sequence.fall(dt)
        t, pos, v = landing_sequence.orient()
        positions.append(pos.copy())
        velocities.append(v.copy())
        time.append(t)
    
    t, pos, v = landing_sequence.orient()
    para_height = np.linalg.norm(pos)-system.radii[1]*1e3
    landing_sequence.deploy_parachute()

    dt = 1
    while np.linalg.norm(pos) > system.radii[1]*1e3:
        landing_sequence.fall(dt)
        t, pos, v = landing_sequence.orient()
        positions.append(pos.copy())
        velocities.append(v.copy())
        time.append(t)
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    time = np.array(time)
    
    landing_sequence.finish_video()
    
    # desired_landing_spot = np.array([radius, 0, 0.44028 * np.pi, 163850])
#     landing_site_position(desired_landing_spot, t[-1]-t[0])
#     plt.figure()
#     plt.scatter(radius/1e3*np.cos(desired_landing_phi), radius/1e3*np.sin(desired_landing_phi), label="desired landing-spot", color="green")
#     plt.plot(positions[:,0]/1e3, positions[:,1]/1e3, label="Lander", color="orange")
#     angle = np.linspace(0, 2*np.pi, 1000, endpoint=False)
#     plt.fill(radius/1e3*np.cos(angle),radius/1e3*np.sin(angle), color="blue", label="Tvekne")
#     plt.axis("equal")
#     plt.title("The landers trajectory and Tvekne")
#     plt.xlabel("x [km]")
#     plt.ylabel("y [km]")
#     plt.legend()
    
    
    print(para_height)

if __name__ == "__main__":
    landing_sequence = initiate_orbit()
    trial_and_error(copy.deepcopy(landing_sequence), 5640, -1000)
    # land4real(landing_sequence, 5640, -1000, 86.13)
