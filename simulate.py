import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# Grid setup
GRID_WIDTH, GRID_HEIGHT = 100, 100  #size of grid
dx = dy = 1.0 # spacial step size between grid points (1 unit apart)
c = 2.0 # wave speed
dt = 0.1 #times step (amount of time wave advances each frame) CFL condition c*(dt/dx) <= 1/sqrt(2)
steps = 500 # num frames

# Wave field arrays
u = np.zeros((GRID_WIDTH, GRID_HEIGHT))      # current
u_new = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # next
u_old = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # previous

# Initial disturbance in the center
u[GRID_WIDTH//2, GRID_HEIGHT//2] = -1

fig, ax = plt.subplots()
im = ax.imshow(u, cmap='magma', vmin=-0.4, vmax=0.4)

def update(frame):
    global u, u_new, u_old
    # finite difference wave equation
    u_new[1:-1,1:-1] = (2*u[1:-1,1:-1] - u_old[1:-1,1:-1] +
                        (c*dt/dx)**2 * (
                            u[2:,1:-1] + u[:-2,1:-1] +
                            u[1:-1,2:] + u[1:-1,:-2] -
                            4*u[1:-1,1:-1]))
    # rotate arrays
    u_old, u, u_new = u, u_new, u_old
    u_smooth = gaussian_filter(u, sigma=1)
    im.set_array(u_smooth)
    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
plt.show()