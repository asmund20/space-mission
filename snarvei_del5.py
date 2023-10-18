from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts

seed = 59529
code_unstable_orbit = 67311

mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [code_unstable_orbit])

################################################################
#              PLACE SPACECRAFT IN UNSTABLE ORBIT              #
################################################################
#                   |      For Part 5      |
#                   ------------------------

"""
DOCUMENTATION

------------------------------------------------------------------------
place_spacecraft_in_unstable_orbit() places the spacecraft in a
randomized elliptical orbit around the specified planet.

Parameters
----------
time  :  float
    The time at which the spacecraft should be placed in orbit, in YEARS
    from the initial system time.

planet_idx  :  int
    The index of the planet that the spacecraft should orbit.

Raises
------
RuntimeError
    When none of the provided codes are valid for unlocking this method.
RuntimeError
    When called before verify_manual_orientation() has been called
    successfully.
------------------------------------------------------------------------

"""

# you can only use this shortcut after successfully calling
# mission.verify_manual_orientation(), as you did to complete Part 4
time = # insert the time you want the spacecraft to be placed in orbit
planet_idx = # insert the index of your destination planet

shortcut.place_spacecraft_in_unstable_orbit(time, planet_idx)

# initiating landing sequence. Documentation on how to use your
# LandingSequence instance can be found here:
#     https://lars-frogner.github.io/ast2000tools/html/classes/ast2000to
#     ols.space_mission.LandingSequence.html#ast2000tools.space_mission.
#     LandingSequence

# USE THE LANDING INSTANCE TO GET A STABLE ORBIT FOR YOUR SPACECRAFT:
land = mission.begin_landing_sequence()
# OBS!
# this can be helpful:
# print()
# land.orient()
# print()