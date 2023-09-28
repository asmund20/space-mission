#=======================#
#   IKKE BRUKT KODEMAL  #
#=======================#

import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
import numpy as np

# calculates the kepler orbit
def calculate_r(f, e, a):
    return a*(1-e**2)/(1+e*np.cos(f))


def main():
    # finner informasjonen som trengs om planetene for å beregne banene
    seed = 59529
    system = SolarSystem(seed)
    f = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    e = system.eccentricities
    a = system.semi_major_axes
    aphelion = system.aphelion_angles

    # til plotting
    colors = ['#cb2121', '#ffa32d', '#ffd700', \
              '#12af83', '#668aff', '#bc7ff7', '#8b1ec4']
    
    # lager et array for hver av planetene med r(f)
    r = list()
    for ei, ai in zip(e, a):
        r.append(calculate_r(f, ei, ai))
    r = np.asarray(r)

    
    for i, (ri, api) in enumerate(zip(r, aphelion)):
        # i plottingen, roterer en banene slik at de stemmer med initialbetingelsene
        plt.plot(ri*np.cos(f + api + np.pi), ri*np.sin(f + api + np.pi), color=colors[i], label=f"planet {i}")

    # plotter stjerna i midten som et punkt
    plt.scatter(0, 0, marker="o", color=[c/255 for c in system.star_color], label='Stellaris Skarsgård')
    plt.ylabel("y [AU]")
    plt.xlabel("x [AU]")
    


if __name__ == "__main__":
    main()
    # har dette her for å kunne bruke plottet i et annet program
    plt.legend()
    plt.axis("equal")
    plt.show()
