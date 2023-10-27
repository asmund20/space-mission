#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

# Men vi har brukt snarvei, siden vi ikke gjorde A og B del 4
# og trengte hjelp til å få sonden i bane rundt Tvekne.

from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from reference_frame import sim_launch
from numba import jit
import numpy as np

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

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



@jit(nopython=True)
def trajectory(initial_time, position, velocity, N_steps, dt):
    """
    Simulates the trajectory from initial_time to (initial_time + N_steps*dt)
    only taking into account the gravitational pull of the planets in Dagobah
    and the star.
    ---------------------------------
    Input: 
        initial_time - time to start trajectory
        position - initial position of probe
        velocity - initial velocity of probe
        N_steps - how many timesteps to simulate trajectory
        dt - timestep

    Returns:
        t - time at the end of trajectory
        position - final position after simulated trajectory
        velocity - final velocity after simulated trajectory
    """

    desired_times = np.linspace(initial_time, initial_time+N_steps*dt, N_steps)
    planet_pos_interp = np.zeros((2, len(pos_planets[0]), N_steps))

    for planet, _ in enumerate(pos_planets[0]):
        planet_pos_interp[0,planet,:] = np.interp(desired_times, known_times, pos_planets[0,planet,:])
        planet_pos_interp[1,planet,:] = np.interp(desired_times, known_times, pos_planets[1,planet,:])

    i = 0
    t = initial_time

    while i < N_steps:
        # Calculate acceleration of probe
        g = -cs.G_sol*star_mass*position/np.linalg.norm(position)**3
        for planet, planet_mass in enumerate(planet_masses):
            g += cs.G_sol*planet_mass*(planet_pos_interp[:,planet,i]-position)/np.linalg.norm(planet_pos_interp[:,planet,i]-position)**3

        # Numerical integration: Euler-Cromer
        velocity += g*dt
        position += velocity*dt

        i += 1
        t += dt

    return t, position, velocity



def get_launch_parameters():
    """
    Estimates the launch parameters based on the intuitive idea
    sketched in the article.
    -------------------------------
    Returns: 
        t0 - launchtime
        phi0 - estimated initial angle
        time - estimated travel duration
    """
    # r0: position Zeron, r1: position Tvekne
    r0, r1 = pos_planets[:,0,:], pos_planets[:,1,:]

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

    time = np.linalg.norm(r1[:,i]-r0[:,i])/abs(vr_rocket)

    return t0, phi0, time



def test(travel_start_time, position, velocity, travel_duration, plot=False, plot_system=False, trajectory_label="Planlagt bane"):
    """
    Tests if preliminary trajectory plan i successful.
    -------------------------------------------------
    Inputs:
        travel_start_time - initial time of trajectory
        position - initial position
        velocity - initial velocity
        travel_duration - travel duration of trajectory
        plot - if True, plots the trajectory and final position of probe
        plot_system - if True, plots planets and star
    Returns:
        position - final position of probe
    """
    N = 1000
    p = np.zeros((N,2))
    p[0] = position
    v = np.zeros((N,2))
    v[0] = velocity
    trajectory_dt = 1e-5
    t = travel_start_time
    for i in range(N-1):
        t, p[i+1], v[i+1] = trajectory(t, p[i], v[i], int(travel_duration/N/trajectory_dt), trajectory_dt)

    position = p[-1]
    velocity = v[-1]
    rocket_from_tvekne = position-pos_planets[:,1,int((t)/1e-4)]
    l = np.linalg.norm(position)*np.sqrt(system.masses[1]/10/system.star_mass)
    in_orbit = np.linalg.norm(rocket_from_tvekne) < l
    rad_vel = np.dot(velocity, position)/np.linalg.norm(position)
    print('#'*25)
    print(f"Close enough to Tvekne for orbit? {in_orbit}")
    print(f"position: {position} AU\nvelocity: {velocity} AU/yr")
    print("Distance from Tvekne/desired distance:", np.linalg.norm(rocket_from_tvekne)/l)
    print(f"Velocity Tvekne: {vel_planets[:,1,int(t/1e-4)]} AU/yr")
    print(f"Final radial velocity of rocket: {rad_vel} AU/yr")
    print('#'*25)

    if plot:
        plt.figure(figsize=(8,8))
        plt.scatter(position[0], position[1], color='green', label="Sonde")
        plt.plot(p[1:,0], p[1:,1], label=trajectory_label, color='green')

    if plot_system:
        plt.scatter(0,0, color='orange', label="Stellaris Skarsgård")
        theta = np.linspace(0,2*np.pi,1000)
        plt.plot(l*np.cos(theta)+pos_planets[0,1,int((t)/1e-4)], l*np.sin(theta)+pos_planets[1,1,int((t)/1e-4)], label="Avstand fra Tvekne < l")
        i_s = int(travel_start_time/1e-4)
        i_f = int(t/1e-4) + 1
        plt.scatter(pos_planets[0,1,i_f-1], pos_planets[1,1,i_f-1], color='blue', label="Tvekne")
        plt.plot(pos_planets[0,0,i_s:i_f], pos_planets[1,0,i_s:i_f], color='red', label="Planetbane Zeron", linestyle="--")
        plt.plot(pos_planets[0,1,i_s:i_f], pos_planets[1,1,i_s:i_f], color='blue', label="Planetbane Tvekne", linestyle="--")
        plt.xlabel('x [AU]', fontsize=12)
        plt.ylabel('y [AU]', fontsize=12)


    return position



def plan_trajectory(plot=False, plot_system=False):
    """
    Plan the preliminary trajectory by adjusting the launch parameters.
    -------------------------------------------------------------------
    Inputs:
        plot - if True, plots the trajectory and final position of probe
        plot_system - if True, plots planets and star
    
    Returns:
        launch_time - time to launch the rocket 
        theta0 - launch angle
        travel_duration - duration of planned trajectory
        endpoint - final position of probe
    """
    # Initial parameters from intuitive idea.
    launch_time, theta, travel_duration = get_launch_parameters()

    # Adaptation of the parameters by trial and error
    launch_time += 0
    theta += 0.111
    travel_duration += 0.8493

    # Simulate launch
    r, vf, r0, theta0 = sim_launch(launch_time, theta)
    travel_start_time = launch_time + launch_duration
    endpoint = test(travel_start_time, r[-1], vf, travel_duration, plot, plot_system)

    return launch_time, theta0, travel_duration, endpoint



def liftoff():
    """
    Actual launch and trajectory from Zeron to Tvekne.
    --------------------------------------------------
    """
    # Plan preliminary trajectory
    time_start_launch, theta0, travel_duration, endpoint = plan_trajectory(plot=True)
    # Simulate launch with said parameters ^
    rocket_positions_during_launch, rocket_velocity_after_launch, _, _ = sim_launch(time_start_launch, theta0)
    fuel_consumption, thrust, fuel = np.load('rocket_specs.npy')

    # Verify successful launch
    mission.set_launch_parameters(thrust, fuel_consumption, fuel, ut.yr_to_s(launch_duration), rocket_positions_during_launch[0], time_start_launch)
    mission.launch_rocket()
    mission.verify_launch_result(rocket_positions_during_launch[-1])

    ### SHORTCUT ###
    sc_position, sc_velocity, sc_motion_angle = shortcut.get_orientation_data()
    mission.verify_manual_orientation(sc_position, sc_velocity, sc_motion_angle)
    ################

    # Start interplanetary travel
    intertravel = mission.begin_interplanetary_travel()

    # Initial coordinates mesured from spacecraft
    it_t, it_pos, it_vel = intertravel.orient()
    traj_pos, traj_vel = it_pos, it_vel

    # Adjust coasttime to fit trajectory
    coasttime = travel_duration/180

    # Calculate timestep of trajectory
    N = 1000
    traj_dt = coasttime/N

    # Desired initial velocity
    desired_velocity = rocket_velocity_after_launch

    # Boost to achieve desired velocity
    dv = desired_velocity - it_vel + (endpoint-it_pos)/travel_duration/10
    intertravel.boost(dv)

    pos = it_pos
    interpositions = [pos]

    t = it_t
    while t < time_start_launch + travel_duration:

        intertravel.coast(coasttime)
        # Expected position and velocity from simulation
        _, traj_pos, traj_vel = trajectory(t,traj_pos,traj_vel,N,traj_dt)
        # Measured coordinates
        t, pos, vel = intertravel.orient()
        interpositions.append(pos)

        # Adjust velocity to stay on track
        dv = traj_vel - vel 
        intertravel.boost(dv)

    interpositions.append(pos)

    # Last attempt at boosting in general direction of Tvekne
    rot_angle = 0.7
    M = np.asarray([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)],
        ])
    dv = np.matmul(M, (endpoint-pos)/coasttime/10)
    intertravel.boost(dv)

    # Drifting some extra to see if we can reach Tvekne
    for _ in range(10):
        intertravel.coast(coasttime*2)
        t, pos, vel = intertravel.orient()
        interpositions.append(pos)

    interpositions = np.array(interpositions)

    print(f"Amount of fuel left in the tank: {intertravel.remaining_fuel_mass} kg")
    print(f"Distance to Tvekne: {np.linalg.norm(interpositions[-1]-pos_planets[:,1,int((t)/1e-4)])}")

    # Plotting
    plt.scatter(0,0, color='orange', label="Stellaris Skarsgård")
    l = np.linalg.norm(interpositions[-1])*np.sqrt(system.masses[1]/10/system.star_mass)
    theta = np.linspace(0,2*np.pi,1000)
    plt.plot(interpositions[1:,0], interpositions[1:,1], color='black', label="Bane uten boost")
    plt.scatter(interpositions[-1,0], interpositions[-1,1], color='black', label='Sonden')
    plt.plot(l*np.cos(theta)+pos_planets[0,1,int((t)/1e-4)], l*np.sin(theta)+pos_planets[1,1,int((t)/1e-4)], label="Avstand fra Tvekne < l")
    i_s = int(it_t/1e-4)
    i_f = int(t/1e-4) + 1
    plt.scatter(pos_planets[0,1,i_f-1], pos_planets[1,1,i_f-1], color='blue', label="Tvekne")
    plt.plot(pos_planets[0,0,i_s:i_f], pos_planets[1,0,i_s:i_f], color='red', label="Planetbane Zeron", linestyle="--")
    plt.plot(pos_planets[0,1,i_s:i_f], pos_planets[1,1,i_s:i_f], color='blue', label="Planetbane Tvekne", linestyle="--")
    plt.xlabel('x [AU]', fontsize=12)
    plt.ylabel('y [AU]', fontsize=12)



def orbit_analysis(pos, vel, land):
    """
    Analyse the final orbit of the spacecraft around Tvekne.
    -------------------------------------------------------
    Inputs:
        pos - position of spacecraft right after injection maneuver
        vel - velocity of spacecraft right after injection maneuver
        land - landing object from ast2000tools
    
    Returns:
        r - distances from Tvekne calculated analytically with Kepler orbit
        f - angles measured from periapsis: [0, 2pi)
        alpha - angle to periapsis from x-axis
        p - h^2/GM use to plot orbit later

    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    e_theta = -np.cross(pos, np.array([0,0,1]))/r
    v_theta = np.dot(vel, e_theta)
    v_r = np.linalg.norm(vel-e_theta*v_theta)
    theta = np.arccos(pos[0]/r)

    print(f'r,v = {r}, {v}')
    print(f'm = {land.mission.spacecraft_mass} kg')
    print(f"e_r prikket med e_theta{np.dot(pos/r, e_theta)}")
    print(f'sqrt(v_r^2 + v_theta^2) = {np.sqrt(v_theta**2 + v_r**2)}')
    print(f'v_theta, v_r = {v_theta}, {v_r}')

    if pos[1] < 0:
        theta = -theta

    m_1 = land.mission.spacecraft_mass
    m_2 = cs.m_sun*system.masses[1]
    M = m_1 + m_2
    mu_hat = m_1*m_2/M
    h = r*v_theta
    p = h**2/M/cs.G
    print(f'p = {p}')
    
    E = 1/2*mu_hat*v**2 - cs.G*M*mu_hat/r
    print(f'E = {E}')

    e = np.sqrt(2*E*p/mu_hat/cs.G/M+1)
    print(f'e = {e}')

    a = p/(1-e**2)
    b = a*np.sqrt(1-e**2)
    P = 2*np.pi*a*b/h
    print(f'a = {a}m\nb = {b}m\nP = {P}s')

    f = np.arccos((p-r)/e/r)
    if v_r < 0:
        f = -f
    alpha = theta-f

    # Kepler orbit
    f = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    r = p/(1+e*np.cos(f))

    print(f'Periapsis: {np.linalg.norm([p/(1+e)*np.cos(alpha),p/(1+e)*np.sin(alpha)])}\n{[p/(1+e)*np.cos(alpha),p/(1+e)*np.sin(alpha)]}')
    print(f'Apoapsis: {np.linalg.norm([p/(1-e)*np.cos(alpha+np.pi),p/(1-e)*np.sin(alpha+np.pi)])}\n{[p/(1-e)*np.cos(alpha+np.pi),p/(1-e)*np.sin(alpha+np.pi)]}')

    # Plotting
    plt.plot(np.cos(f+alpha)*r, np.sin(f+alpha)*r, color='black', linestyle='--', label="Kepler-banen: $r(f) = \\frac{p}{1+e\\cos{f}}$")
    plt.scatter(p/(1+e)*np.cos(alpha), p/(1+e)*np.sin(alpha), color='purple')
    plt.scatter(p/(1-e)*np.cos(alpha+np.pi), p/(1-e)*np.sin(alpha+np.pi), color='purple')
    plt.figtext(0.65, 0.83, 'Apoapsis', fontsize=12)
    plt.figtext(0.25, 0.15, 'Periapsis', fontsize=12)



def stabilize_orbit():
    """
    Stabilize the elliptical orbit of the spacecraft around Tvekne.
    --------------------------------------------------------------
    """
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
    # Position and velocity right after boost
    rf, vf = land.orient()[1:]

    # Let spacecraft orbit for some time in its circular orbit
    pos_after_boost = []
    for _ in range(3*n):
        land.fall(time_step)
        pos = land.orient()[1]
        pos_after_boost.append(pos)

    positions = np.array(positions)
    pos_after_boost = np.array(pos_after_boost)

    # Take picture of Tvekne
    land.look_in_direction_of_planet(1)
    land.take_picture()

    plt.figure(figsize=(8,8))
    plt.plot(positions[:,0], positions[:,1], color='black', linestyle='--', label='Bane før injeksjonsmanøver')
    plt.plot(pos_after_boost[:,0], pos_after_boost[:,1], color='red', label='Bane etter injeksjonsmanøver')
    plt.scatter(positions[-1,0], positions[-1,1], color='black', label='Sonden')
    plt.quiver(positions[-1,0], positions[-1,1], dv_inj[0], dv_inj[1], color='orange', label='Boost $(\\Delta$'+'v'+'$)_{inj}$: injeksjonsmanøveren', scale=2e3, width= 0.005)
    plt.scatter(0,0,label='Tvekne', color='blue')
    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)

    # Analyse and plot orbit
    orbit_analysis(rf,vf,land)



if __name__ == "__main__":
    # liftoff()
    # plan_trajectory(plot=True, plot_system=True)
    stabilize_orbit()
    plt.axis('equal')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()
    
