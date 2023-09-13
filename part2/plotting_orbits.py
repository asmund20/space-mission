import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
import numpy as np

def calculate_r(f, e, a):
    return a*(1-e**2)/(1+e*np.cos(f))


def main():
    seed = 59529
    system = SolarSystem(seed)
    f = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    e = system.eccentricities
    a = system.semi_major_axes
    aphelion = system.aphelion_angles
    
    r = list()
    for ei, ai in zip(e, a):
        r.append(calculate_r(f, ei, ai))
    r = np.asarray(r)

    for i, ri in enumerate(r):
        plt.plot(ri*np.cos(f + aphelion[i]), ri*np.sin(f + aphelion[i]), label=f"planet {i}")

    plt.plot(0, 0, marker="o", color="yellow")
    plt.ylabel("y [AU]")
    plt.xlabel("x [AU]")
    plt.axis("equal")
    plt.legend()

if __name__ == "__main__":
    main()
    plt.show()
