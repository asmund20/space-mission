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
    e_theta = np.array([-r0[1]/np.linalg.norm(r0), r0[0]/np.linalg.norm(r0), 0])
    dv_inj = e_theta*v_stable - v0

    print('#'*20)
    print(f'dv_inj = {dv_inj}')
    print('#'*20)
    land.boost(dv_inj)

    pos_after_boost = []
    for _ in range(n):
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
    print(f'r,v = {r}, {v}')
    print(f'm = {land.mission.spacecraft_mass} kg')
    # e_theta = np.cross(r0, np.array([0,0,1]))/r
    print(f"e_r prikket med e_theta{np.dot(r0/r, e_theta)}")
    v_theta = np.dot(v0, e_theta)
    v_r = np.linalg.norm(v0-e_theta*v_theta)
    print(f'sqrt(v_r^2 + v_theta^2) = {np.sqrt(v_theta**2 + v_r**2)}')
    theta = np.arccos(r0[0]/r)
    print(v_theta, v_r)
    print(np.dot(r0/r, v0))
    print(np.dot(r0/r, r0), r)
    if r0[1] < 0:
        theta = -theta

    m_1 = cs.m_sun*system.masses[1]
    m_2 = cs.m_sun*system.star_mass
    M = m_1 + m_2
    mu_hat = m_1*m_2/M
    h = r*v_theta
    p = h**2/M
    print(p)
    

    T = 1/2*mu_hat*v**2
    U = - cs.G*M*mu_hat/r
    E = 1/2*mu_hat*v**2 - cs.G*M*mu_hat/r
    print(E, T, U)

    e = np.sqrt(2*E*p/mu_hat/M+1)
    print(e)
    a = p/(1-e**2)
    b = a*np.sqrt(1-e**2)
    P = np.sqrt(a**3)

    f = np.arccos((p-r)/e*r)
    if v_r < 0:
        f = -f
    omega = theta-f

    f = np.linspace(0, 2*np.pi, 1000, endpoint=False)

    r = p/(1+e*np.cos(f))

    plt.plot(np.cos(f+omega)*r, np.sin(f+omega)*r, label="nye greier")