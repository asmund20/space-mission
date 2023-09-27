from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import matplotlib.pyplot as plt
import numpy as np
from plotting_orbits import main as plot_orbits

# r i astronomiske enheter
def temp(r, s):
    return s.star_temperature*np.sqrt(s.star_radius*1e3/2/ut.AU_to_m(r))

def main():
    seed = 59529
    system = SolarSystem(seed)

    T = temp(system.semi_major_axes,system)

    for i, t in enumerate(T):
        ikke = " ikke"*(not 260<t<390)
        print(f"Temperatur planet {i} har en overflatetemperatur på omtrent {t:.2f}. Den er{ikke} i habitable zone. {system.types[i]}")

    r = np.linspace(0.0000000008,system.semi_major_axes[6], 1000)
    t = temp(r, system)

    j = np.nonzero(260<t)
    i = np.nonzero(t[j] < 390)

    inner = r[i[0][0]]
    outer = r[i[0][-1]]

    theta = np.linspace(0, 2*np.pi, 1000)

    plt.style.use("dark_background")
    plt.fill(outer*np.cos(theta),outer*np.sin(theta), alpha=0.2, color="white", label="habitable zone")
    plt.fill(inner*np.cos(theta),inner*np.sin(theta), color="black")

    plot_orbits()

    plt.axis("equal")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
