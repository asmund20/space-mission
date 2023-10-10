#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

from matplotlib import pyplot as plt, patches
import numpy as np
import scipy.constants as cs
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
import ast2000tools.utils as utils
import ast2000tools.constants as astconst
from simulate_launch import launch

seed = 59529

system = SolarSystem(seed)
mission = SpaceMission(seed)

#launch time relative to one period for Zeron
# set a positive launch angle for other than directly away from the sun
def sim_launch_relative_period(nperiods, launch_angle=-1):
    a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
    M_s = system.star_mass              # Massen til Stellaris Skarsgard
    M_zeron = system.masses[0]          # Massen til Zeron
    # Perioden til Zeron
    period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(astconst.G_sol*(M_s+M_zeron)))

    return sim_launch(nperiods*period_zeron, launch_angle=launch_angle)

# set a positive launch angle for other than directly away from the sun
# launch time in measured in years
def sim_launch(launch_time, launch_angle=-1):
    # Perioden til Zeron
    a_zeron = system.semi_major_axes[0] # Store halvakse til hjemplaneten Zeron
    M_s = system.star_mass              # Massen til Stellaris Skarsgard
    M_zeron = system.masses[0]          # Massen til Zeron
    period_zeron = np.sqrt((4*np.pi**2 * a_zeron**3)/(astconst.G_sol*(M_s+M_zeron)))

    # unless the launch angle is set, it will be directly away from the star
    if launch_angle < 0:
        launch_angle = 2*np.pi*launch_time/period_zeron

    r1_norm = np.load('rocket_position.npy')
    planet_positions = np.load('positions.npy')
    planet_velocities = np.load('velocities.npy')

    # Rotasjonshastighet til Zeron
    omega = 2*np.pi/(system.rotational_periods[0]*60*60*24)

    dt = 1e-4
    i = int(launch_time/dt)
    # posisjon til Zeron
    r0 = np.zeros((len(r1_norm),2))
    r0[0,0] = utils.AU_to_m(planet_positions[0,0,i])
    r0[0,1] = utils.AU_to_m(planet_positions[1,0,i])

    # hastighet til Zeron
    v0 = np.zeros((len(r1_norm),2))
    v0[0,0] = utils.AU_pr_yr_to_m_pr_s(planet_velocities[0,0,i])
    v0[0,1] = utils.AU_pr_yr_to_m_pr_s(planet_velocities[1,0,i])

    # Posisjonen til raketten relativt til Zeron
    r1 = np.zeros((len(r1_norm),2))
    r1[0] = r1_norm[0]*np.cos(launch_angle), r1_norm[i]*np.sin(launch_angle)

    # Tid
    t = 0
    dt = 1e-3 # s

    # Massen til stjerna Stellaris Skarsgaard
    M = system.star_mass*astconst.m_sun


    for i in range(len(r1_norm)-1):

        # Regner r1 komponenten til posisjonen
        # legger til ønsket vinkel definert fra x-aksen til launch-posisjonen
        r1[i+1] = np.array([r1_norm[i+1]*np.cos(launch_angle + omega*t), r1_norm[i+1]*np.sin(launch_angle + omega*t)])

        # Numerisk integrasjon
        # Akselerasjonen til Zeron
        a = - cs.G*M*r0[i]/np.linalg.norm(r0[i])**3
        v0[i+1] = v0[i]+a*dt
        r0[i+1] = r0[i]+v0[i+1]*dt # Euler-Cromer

        t += dt

    # Konvertere til AU
    r = utils.m_to_AU(r0+r1)
    r0 = utils.m_to_AU(r0)

    # Slutthastighet i AU/yr
    vf = (r[-1]-r[-2])/utils.s_to_yr(dt)

    return r, vf, r0, launch_angle

# ble brukt til testing
def plot_sim(r, vf, r0, plot_zeron=True, launch_angle=""):
    if launch_angle:
        plt.scatter(r[-1,0],r[-1,1], label=f'Sluttposisjonen til raketten, launch angle = {launch_angle/np.pi:.2f} pi')
        plt.plot(r[:,0], r[:,1], 'k', label=f'Banen til raketten, launch angle = {launch_angle/np.pi:.2f} pi')
    else:
        plt.scatter(r[-1,0],r[-1,1], label='Sluttposisjonen til raketten')
        plt.plot(r[:,0], r[:,1], 'k', label='Banen til raketten')

    if plot_zeron:
        plt.suptitle('Banene fulgt av raketten og Zeron i det inertielle referansesytemet', fontsize=24)
        plt.xlabel('x [AU]', fontsize=12)
        plt.ylabel('y [AU]', fontsize=12)
        plt.plot(r0[:,0], r0[:,1], 'k:', label='Banen til Zeron')
        plt.scatter(r0[-1,0],r0[-1,1], label='Sluttposisjonen til Zeron')

# ble brukt til testing
def plotting_og_slikt():
    #r, vf, r0 = sim_launch_relative_period(2.3)
    r, vf, r0, launch_angle = sim_launch(4.1)
    plot_sim(r, vf, r0, launch_angle=launch_angle)
    print(f'Final velocity rocket: {vf} AU/yr')
    print(f'Final position rocket: {r[-1]} AU')
    r, vf, r0, launch_angle = sim_launch(4.1, launch_angle=np.pi)
    plot_sim(r, vf, r0, False, launch_angle=launch_angle)
    print(f'Final velocity rocket: {vf} AU/yr')
    print(f'Final position rocket: {r[-1]} AU')


    plt.scatter(0,0)
    plt.axis("equal")
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()

    plt.show()
def ast_test():
    launch_time = 3
    dt, z, vz, az, mass, fuel, esc_vel, fuel_consumption, thrust = launch()
    r, v, r0, _ = sim_launch(launch_time)

    mission.set_launch_parameters(thrust, fuel_consumption, fuel[0], dt*len(mass), r[0], launch_time)
    mission.launch_rocket()
    mission.verify_launch_result(r[-1])
if __name__ == "__main__":
    ast_test()
    plotting_og_slikt()
