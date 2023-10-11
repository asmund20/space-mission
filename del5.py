from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from reference_frame import sim_launch
from simulate_launch import launch
import sys

def trajectory(initial_time, position, velocity, time, dt):
    seed = 59529
    system = SolarSystem(seed)

    planet_pos = np.load("positions.npy")

    timesteps_planets = len(planet_pos[0,0])
    known_times = np.linspace(0, timesteps_planets*1e-4, timesteps_planets)
    desired_times = np.linspace(initial_time, initial_time+time, int(time/dt))
    planet_pos_interp = np.zeros((2, len(planet_pos[0]), int(time/dt)))

    for planet, _ in enumerate(planet_pos[0]):
        planet_pos_interp[0,planet,:] = np.interp(desired_times, known_times, planet_pos[0,planet,:])
        planet_pos_interp[1,planet,:] = np.interp(desired_times, known_times, planet_pos[1,planet,:])

    i = 0
    t = initial_time
    while t < initial_time+time:
        ...
        g = 0
        for planet, planet_mass in enumerate(system.masses):
            g += cs.G_sol*planet_mass*(planet_pos_interp[:,planet,i]-position)/np.linalg.norm(planet_pos_interp[:,planet,i]-position)**3

        g -= cs.G_sol*system.star_mass*position/np.linalg.norm(position)**3

        velocity += g*dt
        position += velocity*dt

        i += 1
        t += dt

def get_launch_parameters():
    seed = 59529
    system = SolarSystem(seed)
    mission = SpaceMission(seed)
    """
    Returns: t0 - launchtime
            phi0 - initial angle
    """
    # Planet positions
    pos = np.load('positions.npy')
    # r0: position Zeron, r1: position Tvekne
    r0, r1 = pos[:,0,:], pos[:,1,:]

    # When the planets are the closest
    i = np.argmin(np.linalg.norm(r0-r1, axis=0))
    dt = 1e-4
    t0 = i*dt

    # Zeron velocity and coordinates at index i
    v0_i = (r0[:,i+1] - r0[:,i])/dt
    x0_i, y0_i = r0[:,i]
    r0_i = np.linalg.norm(r0[:,i])

    # Tvekne velocity and coordinates at index i
    v1_i = (r1[:,i+1] - r1[:,i])/dt
    x1_i, y1_i = r1[:,i]
    r1_i = np.linalg.norm(r1[:,i])
    
    # Unit vectors for angle phi for the two planets
    phihat0 = np.array([-y0_i/r0_i, x0_i/r0_i])
    phihat1 = np.array([-y1_i/r1_i, x1_i/r1_i])

    # Angular velocity of the planets
    v0_phi = np.dot(v0_i, phihat0)
    v1_phi = np.dot(v1_i, phihat1)

    # Angular velocity of the rocket
    # Meant to match the angular velocity of Tvekne 
    vphi_rocket = v1_phi - v0_phi

    # Final velocity of rocket from simulation
    v_esc = np.linalg.norm(sim_launch(0,0)[1])
    # Radial velocity of rocket
    vr_rocket = np.sqrt(v_esc**2 - vphi_rocket**2)

    # Find angle to Zeron
    phi_i_candidates = np.array([np.arccos(x0_i/r0_i), -np.arccos(x0_i/r0_i)])
    j = np.argmin(abs(y0_i/r0_i - np.sin(phi_i_candidates)))
    phi_i = phi_i_candidates[j]

    # Launch angle
    phi0 = phi_i + np.arctan(vphi_rocket/vr_rocket)

    return t0, phi0
