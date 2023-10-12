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

seed = 59529
system = SolarSystem(seed)
mission = SpaceMission(seed)

code_orientation_data = 9851
shortcut = SpaceMissionShortcuts(mission, [code_orientation_data])

pos_planets = np.load('positions.npy')
rocket_altitude = np.load('rocket_position.npy')

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
    print(f"In orbit around Tvekne? {in_orbit}")
    print(f"position: {position} AU\nvelocity: {velocity} AU/yr")

    rad_vel = np.dot(velocity, position)/np.linalg.norm(position)
    print(f"Final radial velocity of rocket: {rad_vel} AU/yr")

    if plot:
        theta = np.linspace(0,2*np.pi,1000)
        plt.scatter(0,0)
        plt.scatter(position[0], position[1])
        plt.plot(l*np.cos(theta)+pos_planets[0,1,int((t)/1e-4)], l*np.sin(theta)+pos_planets[1,1,int((t)/1e-4)])
        plt.scatter(pos_planets[0,1,int((t)/1e-4)], pos_planets[1,1,int((t)/1e-4)])
        plt.axis('equal')
        plt.show()


def plan_trajectory(plot=False):

    t0, phi, time = get_launch_parameters()
    r, vf, r0, phi0 = sim_launch(t0, phi+0.0169)
    t, position, velocity = trajectory(t0+ut.s_to_yr(1e-3*(len(rocket_altitude)-1)), r[-1], vf, time+0.587)

    rocket_from_tvekne = position-pos_planets[:,1,int((t)/1e-4)]

    l = np.linalg.norm(position)*np.sqrt(system.masses[1]/10/system.star_mass)
    in_orbit = np.linalg.norm(rocket_from_tvekne) < l
    print(f"In orbit around Tvekne? {in_orbit}")
    print(f"position: {position} AU\nvelocity: {velocity} AU/yr")

    rad_vel = np.dot(velocity, position)/np.linalg.norm(position)
    print(f"Final radial velocity of rocket: {rad_vel} AU/yr")

    if plot:
        theta = np.linspace(0,2*np.pi,1000)
        plt.scatter(0,0)
        plt.scatter(position[0], position[1])
        plt.plot(l*np.cos(theta)+pos_planets[0,1,int((t)/1e-4)], l*np.sin(theta)+pos_planets[1,1,int((t)/1e-4)])
        plt.scatter(pos_planets[0,1,int((t)/1e-4)], pos_planets[1,1,int((t)/1e-4)])
        plt.axis('equal')
        plt.show()

    return t0, phi0, t-t0 


def liftoff():
    t0, phi0, time = plan_trajectory()
    r, v, _, _ = sim_launch(t0, phi0)
    fuel_consumption, thrust, rocket_mass, fuel = np.load('rocket_specs.npy')

    # gjør greier i mission som er nødvendige for å kunne få distnces og doppler-skifer
    mission.set_launch_parameters(thrust, fuel_consumption, fuel, 1e-3*(len(rocket_altitude)-1), r[0], t0)
    mission.launch_rocket()
    mission.verify_launch_result(r[-1])

    ### SHORTCUT ###
    position, velocity, motion_angle = shortcut.get_orientation_data()
    mission.verify_manual_orientation(position, velocity, motion_angle)
    ################

    intertravel = mission.begin_interplanetary_travel()
    t, pos, vel = intertravel.orient()
    # traj_pos, traj_vel = r[-1], v
    traj_pos, traj_vel = pos, vel
    dt = time/1000
    test(t, pos, vel, time, plot=True)
    # while t < t0 + time:
    #     intertravel.coast(dt)
    #     _, traj_pos, traj_vel = trajectory(t,traj_pos,traj_vel,dt)
    #     t, pos, vel = intertravel.orient()
    #     dv = traj_vel - vel
    #     intertravel.boost(dv)
    print('Finished!')

if __name__ == "__main__":
    liftoff()
