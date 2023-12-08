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
import os

seed = 59529
system = SolarSystem(seed)
system.verbose = False
mission = SpaceMission(seed)
mission.verbose = False
np.random.seed(seed)

# from part 6
def temperature(r):
    """Temperature as a function of distance from center r"""
    # Mean molecular mass
    mu = 22*cs.m_p
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

# from part 6
def pressure(r):
    """Pressure as a function of distance from center r"""
    # Mean molecular mass
    mu = 22*cs.m_p
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

# from part 6
def density(r):
    """Density as a function of distance from center r"""
    # Mean molecular mass
    mu = 22*cs.m_p
    return pressure(r)*mu/cs.k_B/temperature(r)
    
def initiate_circular_orbit():

    ### SHORTCUT ###
    code_unstable_orbit = 67311
    code_orientation_data = 9851
    shortcut = SpaceMissionShortcuts(mission, [code_orientation_data])
    unstable_orbit = SpaceMissionShortcuts(mission, [code_unstable_orbit])
    ################

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

    return land

def landing_site_position(landing_site, t):
    """Takes a landing position defined by an array with [r, theta, phi, t] and modifies the time-dependent coordinate phi to match the desired time t"""
    omega = 1/system.rotational_periods[1]/60**2/24
    landing_site[2] = landing_site[2] + (t-landing_site[3])*omega
    landing_site[3] = t

def initiate_orbit():
    """Initiates a low circular orbit. Returns the landing sequence"""
    landing_sequence= initiate_circular_orbit()
    landing_sequence.fall(100)
    _, _, v = landing_sequence.orient()

    # slowing down so that the spacecraft will get closer to the planet
    landing_sequence.boost(-0.8*v)

    # time-step in which we check if the spacecrft is closer than the desired distance
    dt = 100 #s
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

    # making sure that the direction of travel stays the same, for relism-purposes
    if np.linalg.norm(dv_inj) > np.linalg.norm(v):
        dv_inj = -e_theta*v_stable-v

    landing_sequence.boost(dv_inj)

    return landing_sequence

def simulate_landing(landing_sequence, deploy_parachute, parachute_area, initiate_at, initiation_boost):
    """Takes a landing sequence, the height to deploy the parachute, thea parachute area and the time and magnitude of the boost to start the landing.
    Does noe change the landing_sequence-instance"""
    landing_sequence = copy.deepcopy(landing_sequence)
    # orbit until the specified time
    landing_sequence.fall(initiate_at)

    t, pos, v = landing_sequence.orient()

    # perform the specified boost. the boost should be a negeative number in order to enter the atmosphere
    v += initiation_boost * v/np.linalg.norm(v)

    area = mission.lander_area
    
    # the gravitational field
    g = lambda r: - cs.G * system.masses[1]*cs.m_sun * r /np.linalg.norm(r)**3

    # the drag force
    drag = lambda v_drag, rho, A: -1/2 * rho * A * np.linalg.norm(v_drag) * v_drag

    # the velocity of the atmosphere, where we assume that angular velocity of the atmosphere is the same as for the planet
    w = lambda r: np.cross(np.array([0, 0, 1/system.rotational_periods[1]/(60**2*24)]), r)

    positions = [pos.copy()]
    velocities = [v.copy()]
    times = [t]
    accelerations = [np.array([0, 0, 0])]
    d = [0]

    dt = 0.1 #s
    deployed_parachute = False
    final_dt = False

    # used to increase dt after the deployment of the parachute
    count = 0

    os.system("clear")

    print("starting integration")
    while np.linalg.norm(pos) > system.radii[1]*1e3:

        # if the lander is at the desired deployment height and the parachute is not already deployed, deploy parachute
        if np.linalg.norm(pos)-system.radii[1]*1e3 <= deploy_parachute and not deployed_parachute:
            area = max(area, parachute_area)
            
            print(f"Deployed parachute at t: {t}, h: {np.linalg.norm(pos)-system.radii[1]*1e3}, v: {np.linalg.norm(v)}")
            deployed_parachute = True
            # we need higher precision here due to to higher force
            dt = 1e-4 #s

        # count steps after the parachute deployed
        if deployed_parachute and not final_dt:
            count += 1

        # if the parachute is deployed and the time step is 1e-4 and the count is big, increase timestep
        if count and count > 1e4 and not final_dt:
            # one second after deploying the parachute, the acceleration should be quite low and we can increase the timestep again
            dt = 1e-2 #s
            final_dt = True

        current_drag = drag(v-w(pos), density(np.linalg.norm(pos)), area)
        
        # if the drag force from the parachute is too great, it breaks
        if np.linalg.norm(current_drag) > 2.5e5 and deployed_parachute:
            raise RuntimeError(f"Parachute broke due to too high of a force of {np.linalg.norm(current_drag)/2.5e5:.6f} times its capacity")

        # the lander acceleration
        a = g(pos) + current_drag/mission.lander_mass
        d.append(np.linalg.norm(current_drag))

        # integration with euler-cromer
        v += a*dt
        pos += v*dt
        t += dt

        positions.append(pos.copy())
        velocities.append(v.copy())
        accelerations.append(a.copy())
        times.append(t)

    return np.array(times), np.array(positions), np.array(velocities), d, np.array(accelerations)

def trial_and_error(landing_sequence, wait_time, boost, desired_landing_spot):
    """Takes a landing sequence a time to wait before boosting, a boost and a spot to aim for.
    Does not change the landing_sequence instance or the landing_spot array"""
    radius = system.radii[1]*1e3
    desired_landing_spot = desired_landing_spot.copy()
    # the analytical expression for the parachute area needed to achieve a soft landing
    A = 2*cs.G*system.masses[1]*cs.m_sun*mission.lander_mass/density(radius)/radius**2/3**2

    t, position, velocity, drag_force, acceleration = simulate_landing(landing_sequence, 1000, A, wait_time, boost)

    # calculates the landing site position at the time of touchdown
    landing_site_position(desired_landing_spot, t[-1])

    final_radial_velocity = np.dot(position[-1], velocity[-1])/np.linalg.norm(position[-1])
    landing_site_phi = np.angle(complex(position[-1,0], position[-1,1]))
    diff_angle = desired_landing_spot[2]-landing_site_phi
    missed_with = radius*diff_angle

    print("Missed desired landing spot with", missed_with/1e3, "km")
    print("Hitting the surface with a radial velocity of", final_radial_velocity, "m/s")
    plot_prelim_traj(t, position, velocity, np.linalg.norm(acceleration, axis=1), desired_landing_spot[2], A)


def plot_prelim_traj(t, position, velocity, acceleration, desired_landing_phi, A):
    """Plotting the height, speed, acceleration and position"""
    radius = system.radii[1]*1e3

    print("parachute_area:", A)
    print("Final acceleration:", acceleration[-1]) 

    # height-plot
    plt.plot(t, np.linalg.norm(position, axis=1)/1e3-radius/1e3)
    plt.title("Høyde over bakken (simulert)", fontsize=12)
    plt.xlabel("Tid [s]", fontsize=12)
    plt.ylabel("Høyde [km]", fontsize=12)

    # speed-plot
    plt.figure()
    plt.plot(t, np.linalg.norm(velocity, axis=1))
    plt.title("Enhetens totalfart (simulert)", fontsize=12)
    plt.xlabel("Tid [s]", fontsize=12)
    plt.ylabel("Fart [m/s]", fontsize=12)

    #acceleration-plot
    plt.figure()
    plt.plot(t, acceleration)
    plt.title("Landingsenhetens akselerasjon (simulert)", fontsize=12)
    plt.xlabel("Tid [s]", fontsize=12)
    plt.ylabel("Akselerasjon [m/s²]", fontsize=12)

    # position-plot
    plt.figure()
    angle = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    plt.fill(radius/1e3*np.cos(angle),radius/1e3*np.sin(angle), color="blue", label="Tvekne")
    plt.scatter(radius/1e3*np.cos(desired_landing_phi), radius/1e3*np.sin(desired_landing_phi), label="Planlagt landingspunkt", color="red")
    plt.plot(position[:,0]/1e3, position[:,1]/1e3, label="Landingsenhetens bane", color="orange")
    plt.axis("equal")
    plt.title("Preliminær bane", fontsize=12)
    plt.xlabel("x [km]", fontsize=12)
    plt.ylabel("y [km]", fontsize=12)
    plt.legend(fontsize=12)

def plot_landing(time, positions, velocities, phi_planned):
    """Plotting the radial velocity, angular velocity and acceleration."""
    radial_velocities = []
    for pos, vel in zip(positions, velocities):
        radial_velocities.append(np.dot(pos,vel)*pos/np.linalg.norm(pos)**2)
    radial_velocities = np.array(radial_velocities)
    v_r = np.linalg.norm(radial_velocities, axis=1)
    r = np.linalg.norm(positions, axis=1)
    
    tangential_velocities = velocities - radial_velocities
    v_t = np.linalg.norm(tangential_velocities, axis=1)
    
    angular_velocities = v_t/np.linalg.norm(positions, axis=1)
    accelerations = np.linalg.norm((velocities[1:]-velocities[:-1]), axis=1)/(time[1:]-time[:-1])
    radial_accelerations = (v_r[1:]-v_r[:-1])/(time[1:]-time[:-1])
    angular_accelerations = (angular_velocities[1:]-angular_velocities[:-1])/(time[1:]-time[:-1])
    
    landing_site_phi = np.angle(complex(positions[-1,0], positions[-1,1]))
    radius = system.radii[1]
    
    plt.figure()
    angle = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    plt.fill(radius*np.cos(angle),radius*np.sin(angle), color="blue", label="Tvekne")
    plt.scatter(radius*np.cos(phi_planned), radius*np.sin(phi_planned), label="Planlagt landingspunkt", color="red")
    plt.plot(positions[:,0]/1e3, positions[:,1]/1e3, label="Landingsenhetens bane", color="orange")
    plt.axis("equal")
    plt.title('Banen fulgt av landingsenheten', fontsize=12)
    plt.xlabel("x [km]", fontsize=12)
    plt.ylabel("y [km]", fontsize=12)
    plt.legend(fontsize=12)
    
    plt.show()
    
    fig, axs = plt.subplots(ncols=2, sharex=True)
    axs[0].plot(time, v_r, label='Radiell fart $v_r(t)$')
    axs[0].set_ylabel('Radiell fart [m/s]', fontsize=12)
    axs[0].legend(fontsize=12)
    
    axs[1].plot(time, angular_velocities-system.rotational_periods[1]/(60**2*24), color='red', label='Rotasjon $\\frac{d\\phi}{dt}-\\Omega$')
    axs[1].set_ylabel('Rotasjonshastighet [s^-1]', fontsize=12)
    axs[1].legend(fontsize=12)
    fig.supxlabel('Tid [s]')
    
    plt.show()
    
    fig, ax = plt.subplots(1)
    ax.plot(r/1e3, v_r, label='Radiell fart $v_r(r)$')
    ax.grid(True)
    ax.invert_xaxis()
    ax.set_xlabel('Avstand $r$ fra sentrum av Tvekne [km]', fontsize=12)
    ax.set_ylabel('Radiell fart [m/s]', fontsize=12)
    ax.legend(fontsize=12)
    
    plt.show()
    
    plt.plot(time[:-1], accelerations, color='orange', label='Enhetens akselerasjon $a(t)$')
    plt.ylabel('Akselerasjon [m/s^2]', fontsize=12)
    plt.xlabel('Tid [s]', fontsize=12)
    plt.legend(fontsize=12)
    
    plt.show()

def land4real(landing_sequence, falltime, initiation_boost, parachute_area, desired_landing_spot):
    desired_landing_spot = desired_landing_spot.copy()
    # Fall until desired launch time
    landing_sequence.fall(falltime)
    t, pos, v = landing_sequence.orient()
    
    # Slow down
    dv = initiation_boost * v/np.linalg.norm(v)
    # Launch lander
    landing_sequence.launch_lander(dv)
    
    # Start video
    landing_sequence.look_in_direction_of_planet(1)
    landing_sequence.start_video()
    
    # Set parachute area
    landing_sequence.adjust_parachute_area(parachute_area)
    
    # Timestep
    dt = 0.1 #s
    # Height to deploy parachute
    h = 1000 #m
    
    positions = []
    velocities = []
    time = []
    
    # Fall until height h over the ground
    while np.linalg.norm(pos) > system.radii[1]*1e3 + h:
        landing_sequence.fall(dt)
        t, pos, v = landing_sequence.orient()
        positions.append(pos.copy())
        velocities.append(v.copy())
        time.append(t)
    
    t, pos, v = landing_sequence.orient()
    parachute_deployment_height = np.linalg.norm(pos)-system.radii[1]*1e3
    landing_sequence.deploy_parachute()

    # Bigger timestep
    dt = 1 #s
    
    # Fall until 500m above ground
    while np.linalg.norm(pos) > system.radii[1]*1e3 + h/2:
        landing_sequence.fall(dt)
        t, pos, v = landing_sequence.orient()
        positions.append(pos.copy())
        velocities.append(v.copy())
        time.append(t)
    
    # Take picture pointing westwards
    pic_angle = np.angle(complex(pos[0], pos[1]))
    landing_sequence.look_in_fixed_direction(azimuth_angle=np.pi/2 - pic_angle)
    landing_sequence.take_picture('landing_pic.xml')
    landing_sequence.look_in_direction_of_planet(1)
    
    # Fall until ground
    while np.linalg.norm(pos) > system.radii[1]*1e3:
        landing_sequence.fall(dt)
        t, pos, v = landing_sequence.orient()
        positions.append(pos.copy())
        velocities.append(v.copy())
        time.append(t)
    
    # Convert to arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    time = np.array(time)
    
    # Plot
    plot_landing(
        time, positions, velocities, desired_landing_spot[2] 
    )

    landing_site_position(desired_landing_spot, time[-1])
    landing_site_phi = np.angle(complex(positions[-1,0], positions[-1,1]))
    diff_angle = desired_landing_spot[2]-landing_site_phi
    # How much we missed by
    missed_with = system.radii[1]*diff_angle

    print("Deployed parachute at", parachute_deployment_height, "m above the surface")
    print("Missed desired landing-spot with", missed_with, "km")
    


if __name__ == "__main__":
    # How long to wait before launch of lander
    wait_time = 5676.98 # s
    # Boost magnitude
    boost = -1000 # m/s
    # Target
    desired_landing_spot = np.array([system.radii[1]*1e3, np.pi/2, 0.44028 * np.pi, 163850])
    # Start orbit
    landing_sequence = initiate_orbit()
    # Simulate landing
    trial_and_error(landing_sequence, wait_time,  boost, desired_landing_spot)
    # Real landing
    land4real(landing_sequence, wait_time, boost, 86.13, desired_landing_spot)
