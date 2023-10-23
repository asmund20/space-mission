from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from reference_frame import sim_launch
#from del4 import triliteration
#from del4 import rel_vel_spacecraft_xy as doppler
from numba import jit
import numpy as np

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

code_unstable_orbit = 67311
code_orientation_data = 9851
shortcut = SpaceMissionShortcuts(mission, [code_orientation_data])
unstable_orbit = SpaceMissionShortcuts(mission, [code_unstable_orbit])

pos_planets = np.load('positions.npy')
vel_planets = np.load("velocities.npy")
rocket_altitude = np.load('rocket_position.npy')
launch_duration = ut.s_to_yr(1e-3*(len(rocket_altitude)))

star_mass = system.star_mass
planet_masses = system.masses

timesteps_planets = len(pos_planets[0,0])
known_times = np.linspace(0, timesteps_planets*1e-4, timesteps_planets)

@jit(nopython=True)
def trajectory(initial_time, position, velocity, N_steps, dt):

    desired_times = np.linspace(initial_time, initial_time+N_steps*dt, N_steps)
    planet_pos_interp = np.zeros((2, len(pos_planets[0]), N_steps))

    for planet, _ in enumerate(pos_planets[0]):
        planet_pos_interp[0,planet,:] = np.interp(desired_times, known_times, pos_planets[0,planet,:])
        planet_pos_interp[1,planet,:] = np.interp(desired_times, known_times, pos_planets[1,planet,:])

    i = 0
    t = initial_time

    while i < N_steps:
        g = -cs.G_sol*star_mass*position/np.linalg.norm(position)**3
        for planet, planet_mass in enumerate(planet_masses):
            g += cs.G_sol*planet_mass*(planet_pos_interp[:,planet,i]-position)/np.linalg.norm(planet_pos_interp[:,planet,i]-position)**3

        velocity += g*dt
        position += velocity*dt

        i += 1
        t += dt

    return t, position, velocity

def get_launch_parameters():
    """
    Returns: t0 - launchtime
            phi0 - initial angle
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

    # t, position, velocity = trajectory(t, position, velocity, time)
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
    print(f"Close enough to Tvekne for orbit? {in_orbit}")
    print(f"position: {position} AU\nvelocity: {velocity} AU/yr")
    print("Distance from Tvekne/desired distance:", np.linalg.norm(rocket_from_tvekne)/l)
    print(f"Velocity Tvekne: {vel_planets[:,1,int(t/1e-4)]} AU/yr")

    rad_vel = np.dot(velocity, position)/np.linalg.norm(position)
    print(f"Final radial velocity of rocket: {rad_vel} AU/yr")

    if plot:
        plt.figure(figsize=(8,8))
        theta = np.linspace(0,2*np.pi,1000)
        plt.scatter(position[0], position[1], color='green', label="Sonde")
        plt.plot(p[1:,0], p[1:,1], label=trajectory_label, color='green')

    if plot_system:
        plt.scatter(0,0, color='orange', label="Stellaris Skarsgård")
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
    launch_time, phi, travel_duration = get_launch_parameters()

    # adaptation of the parameters by trial and error
    launch_time += 0
    phi += 0.111
    travel_duration += 0.8493

    r, vf, r0, phi0 = sim_launch(launch_time, phi)

    travel_start_time = launch_time + launch_duration
    endpoint = test(travel_start_time, r[-1], vf, travel_duration, plot, plot_system)

    return launch_time, phi0, travel_duration, endpoint



def liftoff():
    # phi0 is the launch angle defined from the x-axis.
    time_start_launch, phi0, travel_duration, endpoint = plan_trajectory(plot=True)
    rocket_positions_during_launch, rocket_velocity_after_launch, _, _ = sim_launch(time_start_launch, phi0)
    fuel_consumption, thrust, fuel = np.load('rocket_specs.npy')

    # gjør greier i mission som er nødvendige for å kunne få distnces og doppler-skifer
    mission.set_launch_parameters(thrust, fuel_consumption, fuel, ut.yr_to_s(launch_duration), rocket_positions_during_launch[0], time_start_launch)
    mission.launch_rocket()
    mission.verify_launch_result(rocket_positions_during_launch[-1])

    ### SHORTCUT ###
    sc_position, sc_velocity, sc_motion_angle = shortcut.get_orientation_data()
    mission.verify_manual_orientation(sc_position, sc_velocity, sc_motion_angle)
    ################

    intertravel = mission.begin_interplanetary_travel()
    it_t, it_pos, it_vel = intertravel.orient()
    traj_pos, traj_vel = it_pos, it_vel
    # Adjust coasttime to fit trajectory
    coasttime = travel_duration/180

    N = 1000
    traj_dt = coasttime/N

    desired_position = rocket_positions_during_launch[-1]
    desired_velocity = rocket_velocity_after_launch

    dv = desired_velocity - it_vel + (endpoint-it_pos)/travel_duration/10
    intertravel.boost(dv)
    pos = it_pos
    interpositions = [pos]

    t = it_t
    while t < time_start_launch + travel_duration:
        intertravel.coast(coasttime)
        _, traj_pos, traj_vel = trajectory(t,traj_pos,traj_vel,N,traj_dt)
        t, pos, vel = intertravel.orient()
        interpositions.append(pos)

        dv = traj_vel - vel 

        intertravel.boost(dv)

    interpositions.append(pos)
    rot_angle = 0.7
    M = np.asarray([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)],
        ])
    dv = np.matmul(M, (endpoint-pos)/coasttime/10)
    intertravel.boost(dv)
    for _ in range(10):
        intertravel.coast(coasttime*2)
        t, pos, vel = intertravel.orient()
        interpositions.append(pos)

    interpositions = np.array(interpositions)

    print(f"Amount of fuel left in the tank: {intertravel.remaining_fuel_mass} kg")
    print(f"Distance to Tvekne: {np.linalg.norm(interpositions[-1]-pos_planets[:,1,int((t)/1e-4)])}")

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

def stabilize_orbit():
    time_start_launch, phi0, travel_duration, endpoint = plan_trajectory()
    rocket_positions_during_launch, rocket_velocity_after_launch, _, _ = sim_launch(time_start_launch, phi0)
    fuel_consumption, thrust, fuel = np.load('rocket_specs.npy')

    # gjør greier i mission som er nødvendige for å kunne få distnces og doppler-skifer
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

    positions = []
    time_step = 1000
    n = 100
    for _ in range(n):
        land.fall(time_step)
        pos = land.orient()[1]
        positions.append(pos)
    
    t0, r0, v0 = land.orient()
    v_stable = np.sqrt(cs.G*cs.m_sun*planet_masses[1]/np.linalg.norm(r0))
    e_theta = np.array([r0[1]/np.linalg.norm(r0), -r0[0]/np.linalg.norm(r0), 0])
    dv_inj = -e_theta*v_stable - v0

    land.boost(dv_inj)

    pos_after_boost = []
    for _ in range(20*n):
        land.fall(time_step)
        pos = land.orient()[1]
        pos_after_boost.append(pos)

    positions = np.array(positions)
    pos_after_boost = np.array(pos_after_boost)

    land.look_in_direction_of_planet(1)
    land.take_picture()

    plt.figure(figsize=(8,8))
    #plt.plot(positions[:,0], positions[:,1], color='black', linestyle='--', label='Bane før injeksjonsmanøver')
    plt.plot(pos_after_boost[:,0], pos_after_boost[:,1], color='red', label='Bane etter injeksjonsmanøver')
    plt.scatter(positions[-1,0], positions[-1,1], color='black', label='Sonden')
    #plt.quiver(positions[-1,0], positions[-1,1], dv_inj[0], dv_inj[1], color='orange', label='Boost dv: injeksjonsmanøveren', scale=2e3, width= 0.005)
    plt.scatter(0,0,label='Tvekne', color='blue')
    plt.xlabel('x [m]', fontsize=12)
    plt.ylabel('y [m]', fontsize=12)

    r = np.linalg.norm(r0)
    v = np.linalg.norm(v0)
    v_theta = np.dot(v0, e_theta)
    v_r = v0-e_theta*v_theta
    theta = np.arccos(r0[0]/r)
    if r0[1] < 0:
        theta = -theta

    m_1 = cs.m_sun*system.masses[1]
    m_2 = cs.m_sun*system.star_mass
    M = m_1 + m_2
    mu_hat = m_1*m_2/M
    h = r*v_theta
    p = h**2/M
    

    E = 1/2*mu_hat*v**2 - cs.G*M*mu_hat/r

    e = np.sqrt(2*E*p/mu_hat/M+1)
    a = p/(1-e**2)
    b = a*np.sqrt(1-e**2)
    P = np.sqrt(a**3)



if __name__ == "__main__":
    liftoff()
    # plan_trajectory(plot=True, plot_system=True)
    # stabilize_orbit()
    plt.axis('equal')
    plt.legend(loc='lower left')
    plt.show()
    
