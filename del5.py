from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
from reference_frame import sim_launch
from simulate_launch import launch
import sys
from del4 import triliteration
from del4 import rel_vel_spacecraft_xy as doppler

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

code_orientation_data = 9851
shortcut = SpaceMissionShortcuts(mission, [code_orientation_data])

pos_planets = np.load('positions.npy')
vel_planets = np.load("velocities.npy")
rocket_altitude = np.load('rocket_position.npy')
launch_duration = ut.s_to_yr(1e-3*(len(rocket_altitude)-1))

def trajectory(initial_time, position, velocity, time, dt=1e-5):

    timesteps_planets = len(pos_planets[0,0])
    known_times = np.linspace(0, timesteps_planets*1e-4, timesteps_planets)
    desired_times = np.linspace(initial_time, initial_time+time, int(time/dt))
    planet_pos_interp = np.zeros((2, len(pos_planets[0]), int(time/dt)))

    for planet, _ in enumerate(pos_planets[0]):
        planet_pos_interp[0,planet,:] = np.interp(desired_times, known_times, pos_planets[0,planet,:])
        planet_pos_interp[1,planet,:] = np.interp(desired_times, known_times, pos_planets[1,planet,:])

    i = 0
    t = initial_time

    while i < int(time/dt):
        g = -cs.G_sol*system.star_mass*position/np.linalg.norm(position)**3
        for planet, planet_mass in enumerate(system.masses):
            g += cs.G_sol*planet_mass*(planet_pos_interp[:,planet,i]-position)/np.linalg.norm(planet_pos_interp[:,planet,i]-position)**3

        velocity += g*dt
        position += velocity*dt

        i += 1
        t += dt

    return t, position, velocity



def fuel_consumed(F, consumption, m, dv):
    return consumption*m*dv/F



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

def test(t, position, velocity, time, plot=False):

    t, position, velocity = trajectory(t, position, velocity, time)
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
        theta = np.linspace(0,2*np.pi,1000)
        plt.scatter(0,0, label="Stellaris Skarsgård")
        plt.scatter(position[0], position[1], label="sonde")
        plt.plot(l*np.cos(theta)+pos_planets[0,1,int((t)/1e-4)], l*np.sin(theta)+pos_planets[1,1,int((t)/1e-4)], label="target area")
        plt.scatter(pos_planets[0,1,int((t)/1e-4)], pos_planets[1,1,int((t)/1e-4)], label="Tvekne")
        plt.axis('equal')
        plt.legend()
        plt.show()

def plan_trajectory(plot=False):
    launch_time, phi, travel_duration = get_launch_parameters()
    # adaptation of the parameters

    # jeg testet litt, disse i kommentarene funker relativt bra men tror kanskje vi må booste utover
    #launch_time += -0.01
    #phi += 0.1
    #travel_duration += 0.98

    launch_time += -0.01
    phi += 0.1
    travel_duration += 0.98

    r, vf, r0, phi0 = sim_launch(launch_time, phi)

    travel_start_time = launch_time + launch_duration
    test(travel_start_time, r[-1], vf, travel_duration, plot)

    return launch_time, phi0, travel_duration

def liftoff():
    # phi0 is the launch angle defined from the x-axis.
    time_start_launch, phi0, travel_duration = plan_trajectory()
    rocket_positions_during_launch, rocket_velocity_after_launch, _, _ = sim_launch(time_start_launch, phi0)
    fuel_consumption, thrust, rocket_mass, fuel = np.load('rocket_specs.npy')

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
    dt = travel_duration/1000

    #print(sc_position, sc_velocity)
    #print(rocket_positions_during_launch[-1], rocket_velocity_after_launch)
    #print(it_pos, it_vel)

    desired_position = rocket_positions_during_launch[-1]
    desired_velocity = rocket_velocity_after_launch
    test(it_t, desired_position, desired_velocity, travel_duration)
    #test(it_t, it_pos, it_vel, travel_duration, plot=True)
    while it_t < time_start_launch + travel_duration:
        intertravel.coast(dt)
        _, traj_pos, traj_vel = trajectory(it_t,traj_pos,traj_vel,dt)
        it_t, pos, vel = intertravel.orient()
        dv = traj_vel - vel
        intertravel.boost(dv)
    print('Finished!')

if __name__ == "__main__":
    #liftoff()
    plan_trajectory(plot=True)
