import sys
import os

# Add parent directory (Wave-Simulation) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import simulate

LENGTH = 100
c = 10.0
frames = 500
startingPoints = [([0,10], -1), ([LENGTH-10,LENGTH], 1)]

wave = simulate.Wave_1D(LENGTH, c, LENGTH, frames, startingPoints)