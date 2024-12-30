import numpy as np
import matplotlib.pyplot as plt
from algorithms import rs, gs, wgs, csgs, wcsgs, pics, piwcs


# Physical Constants
lam = 532e-9
f = 30e-2
sub = 1 / 32
trap_number = 50
cuvetteDim = [10e-3, 40e-3, 10e-3]

# Generate original trap locations and shifted locations
x0 = np.linspace(-cuvetteDim[0] / 2, cuvetteDim[0] / 2, trap_number)
y0 = np.linspace(-cuvetteDim[1] / 2, cuvetteDim[1] / 2, trap_number)
z0 = np.zeros_like(y0)

dx = 1e-3
dy = -1e-3
dz = 0

x1 = x0 + dx
y1 = y0 + dy
z1 = z0 + dz

# Generate SLM screen
Nx = 1920
Ny = 1080
total_pixels = Nx * Ny

Lx = 15.36e-3
Ly = 8.64e-3
dx = 8e-6
dy = dx

x = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
y = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)
X, Y = np.meshgrid(x, y)

# Calls WGS algortihm and passes solution to the PICS algorithm
print("Callings WGS algorithm")
mask, phi1, statsWGS = wgs(lam, X, Y, f, x0, y0, z0, 1)

print("\nCalling PICS algorithm")
mask, statsPICS = pics(lam, X, Y, f, x1, y1, z1, sub, 1, phi1)

plt.figure()
plt.imshow(mask, cmap="gray")
plt.title("PICS Phase Mask")
plt.show()
