import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import random

def fill_array_linearly_between(a, i, j):
    per_step = (a[j]-a[i])/(j-i)


    for k in range(i, j):
        a[k] = a[k-1]+per_step

    return a

def light_curve():
    seed = 59529
    random.seed(seed)
    system = SolarSystem(seed)

    star_radius = ut.km_to_AU(system.star_radius)
    planet_radius = ut.km_to_AU(system.radii[6])
    shadow_area_star = np.pi*star_radius**2
    shadow_area_planet = np.pi * planet_radius**2

    phi_hat = np.asarray([-system.initial_positions[1][6], system.initial_positions[0][6]])
    phi_hat = phi_hat/np.linalg.norm(phi_hat)
    v = np.asarray([system.initial_velocities[0][6], system.initial_velocities[1][6]])
    v_phi = np.dot(v, phi_hat)

    print(f"v: {np.linalg.norm(v)} AU/yr\nv_phi: {v_phi} AU/yr")

    distance_to_move = 2*star_radius-4*planet_radius
    time_full_planet_in_front = distance_to_move/v_phi
    time_part_of_planet_in_front = planet_radius/v_phi

    print(f"tid hele planeten er foran stjerna: {time_full_planet_in_front} yr")
    print(f"tid del av planeten er foran stjerna: {time_part_of_planet_in_front} yr")

    flux_star = 1

    flux_star_fully_blocked = (shadow_area_star - shadow_area_planet)/shadow_area_star

    N = 10000
    plot_time = (time_full_planet_in_front+time_part_of_planet_in_front*2)*2

    start_planet_entrance = int(N/10)
    steps_full_planet_in_front = int(time_full_planet_in_front*N/plot_time)
    steps_part_of_planet_in_front = int(time_part_of_planet_in_front*N/plot_time)

    start_fully_blocked = start_planet_entrance+steps_part_of_planet_in_front
    start_planet_exit = start_fully_blocked + steps_full_planet_in_front

    t = np.linspace(0, plot_time, N)
    flux = np.zeros(N)+1


    # setter fluxen når hele planeten er foran
    for i in range(start_fully_blocked, start_planet_exit+1):
        flux[i] = flux_star_fully_blocked

    # fyller inn en lineær funksjon mellom punktene
    fill_array_linearly_between(flux, start_planet_entrance, start_fully_blocked)

    # fyller inn en lineær funksjon mellom punktene
    fill_array_linearly_between(flux, start_planet_exit, start_planet_exit + steps_part_of_planet_in_front)

    # legger til noise
    flux += np.random.normal(0, 1e-4, size=len(flux))

    plt.plot(t, flux)
    plt.xlabel("t [yr]")
    plt.ylabel("relativ flux fra Stellaris Skarsgård")
    plt.show()


if __name__ == "__main__":
    light_curve()

