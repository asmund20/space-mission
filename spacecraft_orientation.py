




def bakugan():
    time_start_launch, phi0, travel_duration, endpoint = plan_trajectory()
    rocket_positions_during_launch, rocket_velocity_after_launch, _, _ = sim_launch(time_start_launch, phi0)
    fuel_consumption, thrust, fuel = np.load('rocket_specs.npy')

    # Verify launch
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

    # Free fall some time before injection maneuver
    time_step = 1000
    n = 1000
    positions = []
    for _ in range(35):
        land.fall(time_step)
        pos = land.orient()[1]
        positions.append(pos)
    
    # Calculates injection maneuver boost
    t0, r0, v0 = land.orient()
    theta = np.angle(complex(r0[0], r0[1]))
    v_stable = np.sqrt(cs.G*cs.m_sun*planet_masses[1]/np.linalg.norm(r0))
    e_theta = np.array([-r0[1]/np.linalg.norm(r0), r0[0]/np.linalg.norm(r0), 0])
    dv_inj = e_theta*v_stable - v0

    print('#'*20)
    print(f'v0 = {v0}')
    print(f'v_stable*e_theta = {e_theta*v_stable}')
    print(f'theta = {theta}')
    print(f'dv_inj = {dv_inj}')
    print('#'*20)

    land.boost(dv_inj)
    # Position and velocity right after boost
    rf, vf = land.orient()[1:]
