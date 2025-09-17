import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# Grid setup
GRID_WIDTH, GRID_HEIGHT = 100, 100  #size of grid
dx = dy = 1.0 # spacial step size between grid points (1 unit apart)
c = 4.0 # wave speed
dt = 0.1 #times step (amount of time wave advances each frame) CFL condition c*(dt/dx) <= 1/sqrt(2)
steps = 500 # num frames

def round_half_up(n):
    return int(n+0.5) if n > 0 else int(n-0.5)

# Wave field arrays
u = np.zeros((GRID_WIDTH, GRID_HEIGHT))      # current
u_new = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # next
u_old = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # previous

disturbancePoints = [(GRID_WIDTH-5, GRID_HEIGHT//2)]

def generateDisturbance():
    global disturbancePoints
    for i in range(len(disturbancePoints)):
        u[disturbancePoints[i][0], disturbancePoints[i][1]] = -2

# two points define wall
walls = [[(60, 0),(61, GRID_WIDTH//2 - 10)], [(60, GRID_WIDTH//2 - 5),(61, GRID_WIDTH//2 + 5)],[(60, GRID_WIDTH//2 + 10),(61, GRID_WIDTH)]]

def generateWalls():
    global walls
    for w in walls:
        u[w[0][0] : w[1][0], w[0][1] : w[1][1]] = 0

# Initial disturbance
generateDisturbance()

#wall mask
mask = np.zeros_like(u, dtype=bool)
for w in walls:
    mask[w[0][0] : w[1][0], w[0][1] : w[1][1]] = True
u_masked = np.ma.masked_array(u, mask=mask)

cmap = plt.cm.magma.copy()
cmap.set_bad(color='#a5ff63')

fig, ax = plt.subplots()
im = ax.imshow(u_masked, cmap=cmap, vmin=-0.4, vmax=0.4)

disturbTime = 0
wavelength = 20
def update(frame):
    global u, u_new, u_old, disturbTime

    #produces waves
    if disturbTime == wavelength:
        generateDisturbance()
        disturbTime = 0
    else:
        disturbTime += 1
    
    # wall
    generateWalls()

    # finite difference wave equation
    u_new[1:-1,1:-1] = (2*u[1:-1,1:-1] - u_old[1:-1,1:-1] + (c*dt/dx)**2 * (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4*u[1:-1,1:-1]))

    # rotate arrays
    u_old, u, u_new = u, u_new, u_old

    #smooth
    u_smooth = gaussian_filter(u, sigma=round_half_up(0.05*wavelength))
    u_masked = np.ma.masked_array(u_smooth, mask=mask)

    im.set_array(u_masked)
    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
plt.show()