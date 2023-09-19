import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
# import ast2000tools.utils as ut
import ast2000tools.constants as cs
import random


def calculate_orbits():
    seed = 59529
    random.seed(seed)
    system = SolarSystem(seed)

    # have checked which planet has the most signifficant pull
    # on the star, that is planet 6.

    # in the reference frame where the star is stationary
    planet_pos_x = system.initial_positions[0, 6]
    planet_pos_y = system.initial_positions[1, 6]
    planet_pos = np.asarray([planet_pos_x, planet_pos_y])

    planet_vel_x = system.initial_velocities[0, 6]
    planet_vel_y = system.initial_velocities[1, 6]
    planet_vel = np.asarray([planet_vel_x, planet_vel_y])

    planet_mass = system.masses[6]
    star_mass = system.star_mass

    # the star is in the origin, and therefore does not need to be included
    # in the sum for the vectors when calculating the CM
    CM = 1/(planet_mass+star_mass)*planet_mass*planet_pos

    print(CM)
    print(planet_pos)

    # will now change from the star-reference-system to the center of
    # mass reference-frame. The new positions for the star and the planet
    # will now be new_pos = old_pos - CM

    planet_pos -= CM
    star_pos = -CM

    # finding the velocities in the new reference-frame
    # star_vel = -planet_vel*planet_mass*(star_mass + planet_mass)
    star_vel = -planet_mass/star_mass*planet_vel
    # planet_vel = planet_vel*(1-planet_mass*(star_mass + planet_mass))

    print(star_vel, planet_vel)
    print(star_pos, planet_pos)

    RUNTIME = 5*30*np.sqrt((4*np.pi**2 * system.semi_major_axes[0]**3)/(cs.G_sol * (star_mass + planet_mass)))
    dt = 1e-4
    N = int(RUNTIME/dt)

    planet_v = np.zeros((N,2))
    planet_p = np.zeros((N,2))
    star_v = np.zeros((N,2))
    star_p = np.zeros((N,2))
    CM = np.zeros((N,2))

    F = np.zeros((N,2))

    planet_v[0] = planet_vel
    planet_p[0] = planet_pos
    star_v[0] = star_vel
    star_p[0] = star_pos
    r = planet_pos-star_pos
    F[0] = -cs.G_sol*planet_mass*star_mass*r/np.linalg.norm(r)**3
    print(cs.G, planet_mass, star_mass, r, np.linalg.norm(r)**3)

    i = 1

    while i < N:
        a_pp = F[i-1]/planet_mass
        a_sp = -F[i-1]/star_mass
        planet_p[i] = planet_p[i-1]+planet_v[i-1]*dt+1/2*a_pp*dt**2
        star_p[i] = star_p[i-1]+star_v[i-1]*dt+1/2*a_sp*dt**2

        r = planet_p[i]-star_p[i]
        F[i] = -cs.G_sol*planet_mass*star_mass*r/np.linalg.norm(r)**3
        a_p = F[i]/planet_mass
        a_s = -F[i]/star_mass

        planet_v[i] = planet_v[i-1] + 1/2*(a_p + a_pp)*dt
        star_v[i] = star_v[i-1] + 1/2*(a_s + a_sp)*dt

        CM[i] = i/(planet_mass + star_mass) * (planet_mass * planet_p[i] + star_mass * star_p[i])

        i += 1

    mu_hat = planet_mass*star_mass/(planet_mass+star_mass)
    #r = planet_p-star_p
    r = np.asarray([np.linalg.norm(p_p-p_s) for p_p, p_s in zip(planet_p, star_p)])
    v = np.asarray([np.linalg.norm(v_p-v_s) for v_p, v_s in zip(planet_v, star_v)])
    E = 1/2*mu_hat*v**2 - cs.G_sol*planet_mass*star_mass/r
    t = np.linspace(0, RUNTIME, N)
    plt.plot(t, E)
    plt.xlabel("t [yr]")
    plt.ylabel("E [m_sun AU^2/yr^2]")
    plt.tight_layout()
    plt.figure()

    plt.plot(star_p[:,0],star_p[:,1])
    #plt.plot(planet_p[:,0], planet_p[:,1])
    plt.plot(CM[:,0], CM[:,1])
    plt.axis("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.tight_layout()
    plt.figure()

    plt.plot(t, [np.linalg.norm(f) for f in F])
    plt.xlabel("t [yr]")
    plt.ylabel("|F| [m_sun AU/yr^2]")
    plt.tight_layout()
    plt.figure()

    v_pec_r = random.uniform(5, 15)
    i = np.pi * (1/3+0.008)
    v_rad = v_pec_r + np.sin(i)*star_v[::4000,0]
    v_rad += np.random.normal(0, (np.max(v_rad)-v_pec_r)/5, size=len(v_rad))
    plt.plot(t[::4000], v_rad)
    plt.xlabel("t [yr]")
    plt.ylabel("radial velocity of the star seen from another system [AU/yr]")
    plt.tight_layout()
    #plt.ylim([0,np.max(v_rad)])
    


    plt.show()


if __name__ == "__main__":
    calculate_orbits()
