import sys
import os

# Add parent directory (Wave-Simulation) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import simulate

GRID_WIDTH = GRID_HEIGHT = 100
dx = dy = 1.0
c = 4.0
frames = 500
#((x,y), wavelength)
disturbancePoints = [((GRID_WIDTH-5, GRID_HEIGHT//2), 20), 
                     ((5, GRID_HEIGHT//2), 40), 
                     ((GRID_WIDTH//2, GRID_HEIGHT-5), 60), 
                     ((GRID_WIDTH//2, 5), 80)]
walls = []

wave = simulate.Wave_2D(GRID_WIDTH, GRID_HEIGHT, dx, dy, c, frames, disturbancePoints, walls)