import numpy as np
import matplotlib.pyplot as plt
from cpwllib.implementation import N_from_target_error

target_error = 0.001
N = N_from_target_error(target_error)

def g_plus(points, N):
    x = points[:, 0]
    y = points[:, 1]
    return np.max(
        np.array(
            [(2*j+1)*x/(4*N) + (2*j+1)*y/(4*N) - (j*(j+1))/(4*N**2) for j in range(2*N)]
        ),
        axis=0)

def g_minus(points, N):
    x = points[:, 0]
    y = points[:, 1]
    return np.max(
        np.array(
            [-(2*k+1)*x/(4*N) + (2*k+1)*y/(4*N) - (k*(k+1))/(4*N**2) for k in range(-N, N)]
        ),
        axis=0)

# 1D points between 0 and 1
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)

# create a 2D grid
X, Y = np.meshgrid(x, y)

# stack into coordinate pairs
points = np.column_stack([X.ravel(), Y.ravel()])

Z = (g_plus(points, N) - g_minus(points, N)).reshape(-1, len(x))

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
fig.colorbar(surf, shrink=0.5, aspect=5)

fig.savefig("figure-maris-equations.pdf", dpi=300)
fig.savefig("figure-maris-equations.png", dpi=300)

plt.close()
print("figure-maris-equations saved successfully")
