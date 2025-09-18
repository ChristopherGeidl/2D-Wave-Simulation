import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import sys

def round_half_up(n):
    return int(n+0.5) if n > 0 else int(n-0.5)

class Wave_1D:
    def __init__(self, length, speed, nx, frames, startingPoints):
        self.length = length
        self.c = speed
        self.nx = nx #number of discrete points along length
        self.dx = self.length / (self.nx - 1) #distance between points
        self.frames = frames
        self.dt = 0.001

        self.x = np.linspace(0, self.length, self.nx)

        self.u = np.zeros(self.nx)
        self.u_prev = np.zeros(self.nx)
        self.u_next = np.zeros(self.nx)

        #Initial condition:
        ylim = 1
        for p in startingPoints:
            self.u[p[0][0]:p[0][1]] = p[1]
            if abs(p[1]) > ylim:
                ylim = abs(p[1])
        self.u_prev[:] = self.u[:]  # Assume zero initial velocity

        #plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.x, self.u)
        self.ax.set_ylim(-ylim, ylim)

        #run
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, interval=self.dt*1000, blit=True)
        plt.show()
    def update(self, frame):
        for i in range(1, self.nx - 1):
            self.u_next[i] = (2 * self.u[i] - self.u_prev[i] +
                        (self.c * self.dt/self.dx)**2 * (self.u[i+1] - 2*self.u[i] + self.u[i-1]))

        # Apply boundaries
        self.u_next[0] = 0
        self.u_next[-1] = 0

        # Swap arrays
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev

        #smooth
        self.u = gaussian_filter(self.u, sigma=1)

        self.line.set_ydata(self.u)
        return [self.line]

class Wave_2D:
    def __init__(self, grid_width, grid_height, dx, dy, speed, frames, disturbancePoints, walls):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.dx = dx #spacial step size between grid points
        self.dy = dy
        self.c = speed #wave speed
        self.dt = 0.1 #times step (amount of time wave advances each frame) CFL condition c*(dt/dx) <= 1/sqrt(2)
        self.frames = frames
        self.disturbancePoints = disturbancePoints #list of ((x,y), wavelength)
        self.walls = walls #list of ((x1, y1), (x2, y2)) must be horizontal or vertical

        self.u = np.zeros((self.grid_width, self.grid_height))      # current
        self.u_new = np.zeros((self.grid_width, self.grid_height))  # next
        self.u_old = np.zeros((self.grid_width, self.grid_height))  # previous

        # Initial disturbances & find minimum wavelength
        self.min_wavelength = sys.maxsize #for blur calculation in update
        for i in range(len(self.disturbancePoints)):
            self.u[self.disturbancePoints[i][0][0], self.disturbancePoints[i][0][1]] = -2
            if self.disturbancePoints[i][1] < self.min_wavelength:
                self.min_wavelength = self.disturbancePoints[i][1]

        #wall mask
        self.mask = np.zeros_like(self.u, dtype=bool)
        for w in self.walls:
            self.mask[w[0][0] : w[1][0], w[0][1] : w[1][1]] = True
        self.u_masked = np.ma.masked_array(self.u, mask=self.mask)

        self.cmap = plt.cm.magma.copy()
        self.cmap.set_bad(color='#a5ff63') #wall color

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.u_masked, cmap=self.cmap, vmin=-0.4, vmax=0.4)

        self.currentFrame = 0 #tracks frames so that distrubances are on time with wavelength

        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, interval=30, blit=True)
        plt.show()
    def generateWalls(self):
        for w in self.walls:
            self.u[w[0][0] : w[1][0], w[0][1] : w[1][1]] = 0
    def update(self, frame):
        for i in range(len(self.disturbancePoints)):
            if self.currentFrame % self.disturbancePoints[i][1] == 0:#current frame is a factor of wavelength
                self.u[self.disturbancePoints[i][0][0], self.disturbancePoints[i][0][1]] = -2
        self.currentFrame += 1

        self.generateWalls()

        #finite difference wave equation
        self.u_new[1:-1,1:-1] = (2*self.u[1:-1,1:-1] - self.u_old[1:-1,1:-1] + (self.c*self.dt/self.dx)**2 * (self.u[2:,1:-1] + self.u[:-2,1:-1] + self.u[1:-1,2:] + self.u[1:-1,:-2] - 4*self.u[1:-1,1:-1]))

        # rotate arrays
        self.u_old, self.u, self.u_new = self.u, self.u_new, self.u_old

        #smooth
        u_smooth = gaussian_filter(self.u, sigma=round_half_up(0.05*self.min_wavelength))
        self.u_masked = np.ma.masked_array(u_smooth, mask=self.mask)

        self.im.set_array(self.u_masked)
        return [self.im]