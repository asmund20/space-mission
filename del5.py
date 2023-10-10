from ast2000tools.solar_system import SolarSystem
import ast2000tools.utils as ut
import ast2000tools.constants as cs
import matplotlib.pyplot as plt
import numpy as np
from ast2000tools.space_mission import SpaceMission
from reference_frame import sim_launch
from simulate_launch import launch
import sys

def print_info():
    seed = 59529
    system = SolarSystem(seed)
    mission = SpaceMission(seed)


