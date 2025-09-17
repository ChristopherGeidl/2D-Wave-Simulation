import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

def round_half_up(n):
    return int(n+0.5) if n > 0 else int(n-0.5)

class Wave_2D:
    def __init__(self, grid_width, grid_height, dx, dy, speed, wavelength, frames, disturbancePoints, walls):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.dx = dx #spacial step size between grid points
        self.dy = dy
        self.c = speed #wave speed
        self.wavelength = wavelength
        self.dt = 0.1 #times step (amount of time wave advances each frame) CFL condition c*(dt/dx) <= 1/sqrt(2)
        self.frames = frames
        self.disturbancePoints = disturbancePoints
        self.walls = walls

        self.u = np.zeros((self.grid_width, self.grid_height))      # current
        self.u_new = np.zeros((self.grid_width, self.grid_height))  # next
        self.u_old = np.zeros((self.grid_width, self.grid_height))  # previous

        # Initial disturbance
        self.generateDisturbance()

        #wall mask
        self.mask = np.zeros_like(self.u, dtype=bool)
        for w in self.walls:
            self.mask[w[0][0] : w[1][0], w[0][1] : w[1][1]] = True
        self.u_masked = np.ma.masked_array(self.u, mask=self.mask)

        self.cmap = plt.cm.magma.copy()
        self.cmap.set_bad(color='#a5ff63')

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.u_masked, cmap=self.cmap, vmin=-0.4, vmax=0.4)

        self.disturbTime = 0

        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, interval=30, blit=True)
        plt.show()
    def generateDisturbance(self):
        for i in range(len(self.disturbancePoints)):
            self.u[self.disturbancePoints[i][0], self.disturbancePoints[i][1]] = -2
    def generateWalls(self):
        for w in self.walls:
            self.u[w[0][0] : w[1][0], w[0][1] : w[1][1]] = 0
    def update(self, frame):
        if self.disturbTime == self.wavelength:
            self.generateDisturbance()
            self.disturbTime = 0
        else:
            self.disturbTime += 1

        self.generateWalls()

        #finite difference wave equation
        self.u_new[1:-1,1:-1] = (2*self.u[1:-1,1:-1] - self.u_old[1:-1,1:-1] + (self.c*self.dt/self.dx)**2 * (self.u[2:,1:-1] + self.u[:-2,1:-1] + self.u[1:-1,2:] + self.u[1:-1,:-2] - 4*self.u[1:-1,1:-1]))

        # rotate arrays
        self.u_old, self.u, self.u_new = self.u, self.u_new, self.u_old

        #smooth
        u_smooth = gaussian_filter(self.u, sigma=round_half_up(0.05*self.wavelength))
        self.u_masked = np.ma.masked_array(u_smooth, mask=self.mask)

        self.im.set_array(self.u_masked)
        return [self.im]