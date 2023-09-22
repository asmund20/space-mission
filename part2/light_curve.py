import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import random

# funksjon som lager en lineær funksjon mellom to indekser i et array
def fill_array_linearly_between(a, i, j):
    per_step = (a[j]-a[i])/(j-i)


    for k in range(i+1, j):
        a[k] = a[k-1]+per_step

def light_curve():
    # antar her at start-posisjonen og start-hastigheten
    # er der en ser planeten skygge for stjerna fra et annet system
    seed = 59529
    random.seed(seed)
    system = SolarSystem(seed)

    star_radius = ut.km_to_AU(system.star_radius)
    planet_radius = ut.km_to_AU(system.radii[6])

    # arealet til en disk gjennom sentrum
    shadow_area_star = np.pi*star_radius**2
    shadow_area_planet = np.pi * planet_radius**2

    # finner enhetsvektoren for vinkel
    phi_hat = np.asarray([-system.initial_positions[1][6], system.initial_positions[0][6]])
    phi_hat = phi_hat/np.linalg.norm(phi_hat)

    # finner hastigheten i phi-retning/normalt på synslinjen
    v = np.asarray([system.initial_velocities[0][6], system.initial_velocities[1][6]])
    v_phi = np.dot(v, phi_hat)

    print(f"v: {np.linalg.norm(v)} AU/yr\nv_phi: {v_phi} AU/yr")

    # finner avstanden planeten beveger på seg mens hele
    # planeten er foran stjerna
    distance_to_move = 2*star_radius-4*planet_radius
    # finner tiden hele planeten er foran stjerna
    time_full_planet_in_front = distance_to_move/v_phi
    # finner tiden en del av planeten er foran stjerna
    # i hver ende
    time_part_of_planet_in_front = 2*planet_radius/v_phi

    print(f"tid hele planeten er foran stjerna: {time_full_planet_in_front} yr")
    print(f"tid del av planeten er foran stjerna: {time_part_of_planet_in_front} yr")

    # opererer med relativ mottatt flux
    flux_star = 1

    # antar at fluxen motatt fra skyggearealet til stjerna er konstant
    # og finner slik den relative fluxen når hele planeten er foran stjerna
    flux_star_fully_blocked = (shadow_area_star - shadow_area_planet)/shadow_area_star

    # lengden på arrayene
    N = 10000
    # en passende lengde å plotte
    plot_time = (time_full_planet_in_front+time_part_of_planet_in_front*2)*2

    # indexen der planeten begynner å skygge for stjerna
    start_planet_entrance = int(N/10)
    # antall punkter i arrayet der hele planeten er foran
    steps_full_planet_in_front = int(time_full_planet_in_front*N/plot_time)
    # antall punkter i arrayet der del av planeten er foran
    steps_part_of_planet_in_front = int(time_part_of_planet_in_front*N/plot_time)

    # første index der hele planeten er foran stjerna
    start_fully_blocked = start_planet_entrance+steps_part_of_planet_in_front
    # indexen der planeten begynner å havne forbi stjerna
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
