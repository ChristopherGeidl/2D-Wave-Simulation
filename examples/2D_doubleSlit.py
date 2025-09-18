import sys
import os

# Add parent directory (Wave-Simulation) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import simulate

GRID_WIDTH = GRID_HEIGHT = 100
dx = dy = 1.0
c = 4.0
wavelength = 20
frames = 500
disturbancePoints = [(GRID_WIDTH-5, GRID_HEIGHT//2)]

# two points define wall
walls = [[(60, 0),(61, GRID_WIDTH//2 - 10)], [(60, GRID_WIDTH//2 - 5),(61, GRID_WIDTH//2 + 5)],[(60, GRID_WIDTH//2 + 10),(61, GRID_WIDTH)]]

wave = simulate.Wave_2D(GRID_WIDTH, GRID_HEIGHT, dx, dy, c, wavelength, frames, disturbancePoints, walls)