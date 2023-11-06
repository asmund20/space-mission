import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import ast2000tools.constants as cs 
import ast2000tools.utils as ut
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from reference_frame import sim_launch
from del5 import plan_trajectory
from numba import jit
import tqdm
import copy
import os

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)
np.random.seed(seed)

vmax_c = 1e4/cs.c

spectral_lines = {
    632: 'O2', 690: 'O2', 760: 'O2', 720: 'H2O',
    820: 'H2O', 940: 'H2O', 1400: 'CO2', 1600: 'CO2',
    1660: 'CH4', 2200: 'CH4', 2340: 'CO', 2870: 'N2O'
}

gasses = {
    'O2': {'name': 'Oxygen', 'A': 32}, 'H2O': {'name': 'Water vapor', 'A': 18},
    'CO2': {'name': 'Carbon dioxide', 'A': 44}, 'CH4': {'name': 'Methane', 'A': 16},
    'CO': {'name': 'Carbon monoxide', 'A': 28}, 'N2O': {'name': 'Nitrous dioxide', 'A': 44}
}



@jit(nopython=True)
def chi_sqr(f, lmbda, lmbda0, sigma, doppler, temp, F_min, m):
    std = (lmbda0/cs.c)*np.sqrt(cs.k_B*temp/m)
    F = 1 + (F_min-1)*np.exp(-((lmbda+doppler-lmbda0)**2)/(2*std**2))
    
    df = len(F)
    chi_squared = np.sum(((f-F)/sigma)**2)
    
    return chi_squared



@jit(nopython=True)
def chi_squared_test(lmbda, lmbda0, idx_lmb0, flux, sigma, m):

    N = int(lmbda0*vmax_c/dlmbda)

    chi_values = []
    param = []
    for doppler in np.linspace(-vmax_c*lmbda0, vmax_c*lmbda0,100):
        for temp in range(150,451,5):
            for F_min in np.linspace(0.6, 1.0, 100):
                chisqr = chi_sqr(
                    flux[idx_lmb0-N:idx_lmb0+N], lmbda[idx_lmb0-N:idx_lmb0+N],
                    lmbda0, sigma[idx_lmb0-N:idx_lmb0+N], doppler, temp, F_min, m
                )
                chi_values.append(chisqr)
                param.append([doppler, temp, F_min])
    return chi_values, param



def atmosphere_chem_comp(lmbda, flux, sigma):
    
    print('Finner parametre ved chi^2-test:')
    parameters = []
    for lmbda0 in tqdm.tqdm(spectral_lines.keys()):
        gas = spectral_lines[lmbda0]
        gas_info = gasses[gas]
        m = gas_info['A']*cs.m_p
        i = np.argmin(abs(lmbda-lmbda0))
        chi_values, param = chi_squared_test(lmbda, lmbda0, i, flux, sigma, m)
        i_min = np.argmin(chi_values)
        
        parameters.append(param[i_min])

    return parameters



def plot_model_over_data(flux, lmbda, dlmbda, parameters):
    print('*'*90)
    print('| Gas                    | lambda0  [nm]  | dlambda [nm]  | Temperature [K]  | F_min     |')
    print('*'*90)

    j = 0
    fig, axs = plt.subplots(nrows=3, ncols=4)
    ymin, ymax = 0.5, 1.5
    for lmbda0 in spectral_lines.keys():
    
        i = np.argmin(abs(lmbda-lmbda0))
        N = int(lmbda0*vmax_c/dlmbda)

        doppler, temp, F_min = parameters[j]
        gas = spectral_lines[lmbda0]
        gas_info = gasses[gas]
        m = gas_info['A']*cs.m_p
        name = gas_info['name']
    
        print(f'| {name:<16} ({gas:<3}) | {lmbda0:<15.2f}| {doppler:<14.5f}| {temp:<17}| {F_min:<10.5f}|')
    
        std = (lmbda0/cs.c)*np.sqrt(cs.k_B*temp/m)
        F = 1 +(F_min-1)*np.exp(-((lmbda[i-N:i+N]+doppler-lmbda0)**2)/(2*std**2))
        axs[j//4, j%4].set_ylim(ymin, ymax)
        axs[j//4, j%4].plot(lmbda[i-N:i+N], flux[i-N:i+N])
        axs[j//4, j%4].plot(lmbda[i-N:i+N], F, color='red', label=f'{gas}: {lmbda0}nm')
        axs[j//4, j%4].legend(loc='upper right')

        j += 1
    
    print('*'*90)
    for ax in axs.flat:
        ax.label_outer()
    fig.supxlabel('Bølgelengde $\\lambda$ [nm]')
    fig.supylabel('Normalisert fluks $F$')
    plt.show()



def plot_sigma(sigma, lmbda, dlmbda):
    fig, axs = plt.subplots(nrows=3, ncols=4)
    ymin, ymax = min(sigma), max(sigma)
    j = 0
    for lmbda0 in spectral_lines.keys():
        gas = spectral_lines[lmbda0]
        i = np.argmin(abs(lmbda-lmbda0))
        N = int(lmbda0*vmax_c/dlmbda)
        
        axs[j//4, j%4].set_ylim(ymin, ymax)
        axs[j//4, j%4].plot(lmbda[i-N:i+N], sigma[i-N:i+N], label=f'$\\sigma_i$ rundt $\\lambda_0$ = {lmbda0}nm')
        axs[j//4, j%4].legend(loc='upper right')
                
        j += 1
    
    for ax in axs.flat:
        ax.label_outer()
    fig.supxlabel('Bølgelengde $\\lambda$ [nm]')
    fig.supylabel('Standard avvik for støyet $\\sigma_i$')
    plt.show()


def temperature(r):
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    T0 = 271
    rho0 = system.atmospheric_densities[1]
    r0 = system.radii[1]*1e3
    M_T = system.masses[1]*cs.m_sun
    gamma = 1.4

    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    if r > r_iso:
        T = T0/2
    else:
        T = T0 - (gamma-1)/gamma * mu*cs.G*M_T/cs.k_B * (1/r0 - 1/r)
    
    return T

def pressure(r):
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    T0 = 271
    rho0 = system.atmospheric_densities[1]
    r0 = system.radii[1]*1e3
    p0 = rho0*cs.k_B*T0/mu
    M_T = system.masses[1]*cs.m_sun
    gamma = 1.4
    
    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    if r > r_iso:
        p_iso = p0 * (T0/temperature(r_iso))**(gamma/(1-gamma))
        p = p_iso * np.exp(-(2*mu*cs.G*M_T)/cs.k_B/T0 * (1/r_iso - 1/r))
    else:
        p = p0 * (T0/temperature(r))**(gamma/(1-gamma))
    
    return p

def density(r):
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    return pressure(r)*mu/cs.k_B/temperature(r)
    

def plot_temp_density(altitude, N):
    r0 = system.radii[1]*1e3
    r = np.linspace(r0, r0+altitude, N)
    temp, dens = np.zeros(N), np.zeros(N)
    
    for i in range(N):
        temp[i] = temperature(r[i])
        dens[i] = density(r[i])
    
    T0 = 271
    gamma = 1.4
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    M_T = system.masses[1]*cs.m_sun
    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    fig, axs = plt.subplots(2, sharex=True)
    
    axs[0].set_xlim(r[0],r[-1])
    axs[0].set_ylim(0,280)
    axs[0].plot(r, temp, label='$T(r)$')
    axs[0].plot([r_iso, r_iso],[0,T0], linestyle='--', color='black', label='$r_{iso}$')
    axs[0].grid(visible=True)
    axs[0].legend(fontsize=12)

    axs[1].set_xlim(r[0],r[-1])
    axs[1].plot(r, dens, color='red', label='$\\varrho (r)$')
    axs[1].plot([r_iso, r_iso],[0,dens[0]], linestyle='--', color='black', label='$r_{iso}$')
    axs[1].grid(visible=True)
    axs[1].legend(fontsize=12)
    
    plt.show()


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

    land.boost(dv_inj)

    return land, pos_planets


# calculates the spherical coordinate for a point on Tvekne's surface.
# takes the current position (theta, phi) and the time after which to return the position.
# returns: theta, phi
def landing_site_position(theta, phi, time):
    omega = 1/system.rotational_periods[1]/60**2/24
    return theta, phi+time*omega


def verify_not_in_atmosphere(landing_sequence):
    fall_time = 1e7

    landing_sequence = copy.deepcopy(landing_sequence)

    landing_sequence.fall(fall_time)


def look_for_landingspot():
    landing_sequence, planet_positions = initiate_circular_orbit()
    t, p, v = landing_sequence.orient()
    landing_sequence.fall(100)
    t, p, v = landing_sequence.orient()

    # slowing down in order to get an orbit that is as close to the planet
    # as possible without entering the atmosphere
    #landing_sequence.boost(-0.7268468*v)
    landing_sequence.boost(-0.8*v)

    dt = 100
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

    if np.linalg.norm(dv_inj) > np.linalg.norm(v):
        dv_inj = -e_theta*v_stable-v

    landing_sequence.boost(dv_inj)

    landing_sequence.fall(250)
    landing_sequence.look_in_direction_of_planet(1)
    landing_sequence.take_picture("landing_site.xml")
    landing_pos_discovery_time, landing_pos_discovery_pos, _ = landing_sequence.orient()
    landing_pos_discovery_radius = np.linalg.norm(landing_pos_discovery_pos)

    os.system("clear")

    print(f"Distance to Tvekne's center: {landing_pos_discovery_radius:.5e} m")
    print(f"Distance to Tvekne's surface: {landing_pos_discovery_radius-system.radii[1]*1000:.5e} m")
    print(f"Landing site coordinates: r = planet_radius, theta = 0, phi = {np.angle(complex(landing_pos_discovery_pos[0], landing_pos_discovery_pos[1]))/np.pi} pi, t = {landing_pos_discovery_time}")

look_for_landingspot()

lmbda, flux = np.load("spectrum_644nm_3000nm.npy")[:,0], np.load("spectrum_644nm_3000nm.npy")[:,1]
sigma = np.load("sigma_noise.npy")[:,1]
dlmbda = (lmbda[-1]-lmbda[0])/len(lmbda)

parameters = atmosphere_chem_comp(lmbda, flux, sigma)
plot_model_over_data(flux, lmbda, dlmbda, parameters)
plot_sigma(sigma, lmbda, dlmbda)

altitude = 0.8e5
N = 10**4
