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
def landing_site_position(theta, phi, time):
    omega = 1/system.rotational_periods[1]/60**2/24
    return theta, phi+time*omega


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
def simulate_landing(landing_sequence, deploy_parachute, parachute_area, initiate_at, initiation_boost, initiation_boost_direction):
    landing_sequence = copy.deepcopy(landing_sequence)
    landing_sequence.fall(initiate_at)

    t, pos, v = landing_sequence.orient()

    initiation_boost_angle = np.angle(complex(pos[0], pos[1]))
    print(v)
    print(initiation_boost * np.array([np.cos(initiation_boost_angle), np.sin(initiation_boost_angle), 0]))
    print(pos)
    v += initiation_boost * np.array([np.cos(initiation_boost_angle), np.sin(initiation_boost_angle), 0])

    area = mission.lander_area
    
    g = lambda r: - cs.G * system.masses[1]*cs.m_sun * r /np.linalg.norm(r)**3

    drag = lambda r, v_drag, rho, A: -1/2 * rho * A * np.linalg.norm(v_drag) * v_drag

    w = lambda r: np.cross(np.array([0, 0, 1/system.rotational_periods[1]/(60**2*24)]), r)

    distances = [np.linalg.norm(pos)]
    speeds = [np.linalg.norm(v)]
    positions = [pos.copy()]
    velocities = [v.copy()]
    times = [t]
    d = [0]

    dt = 0.1
    deployed_parachute = False
    #deployed_parachute = True

    print("starting integration")
    while np.linalg.norm(pos) > system.radii[1]*1e3:

        if np.linalg.norm(pos)-system.radii[1]*1e3 <= deploy_parachute and not deployed_parachute:
            area = max(area, parachute_area)
            print(f"Deployed parachute at t: {t}, h: {np.linalg.norm(pos)-system.radii[1]*1e3}, v: {v}")
            deployed_parachute = True

        a = g(pos) + drag(pos, v-w(pos), density(np.linalg.norm(pos)), area)/mission.lander_mass
        d.append(np.linalg.norm(drag(pos, v-w(pos), density(np.linalg.norm(pos)), parachute_area)))

        v += a*dt
        pos += v*dt
        t += dt

        distances.append(np.linalg.norm(pos))
        speeds.append(np.linalg.norm(v))
        positions.append(pos.copy())
        velocities.append(v.copy())
        times.append(t)

    return np.array(times), np.array(distances), np.array(positions), np.array(speeds), np.array(velocities), d

r = system.radii[1]*1e3
A = 2*cs.G*system.masses[1]*cs.m_sun*mission.lander_mass/density(r)/r**2/3**2
#A = 10

t, r, p, s, v, d = simulate_landing(initiate_orbit(), 1000, A, 0, 500, np.pi/2)
print("parachute_area: ", A)

#def simulate_landing(landing_sequence, deploy_parachute, parachute_area, initiate_at, initiation_boost, initiation_boost_direction):

plt.plot(t, r-system.radii[1]*1e3)
plt.figure()
plt.plot(t, s)
#plt.plot(p[:,0], p[:,1])
plt.figure()
plt.plot(p[:,0], p[:,1])
angle = np.linspace(0, 2*np.pi, 1000, endpoint=False)
plt.plot(system.radii[1]*1e3*np.cos(angle), system.radii[1]*1e3*np.sin(angle))
plt.axis("equal")
plt.figure()
plt.plot(t, d)


plt.show()
