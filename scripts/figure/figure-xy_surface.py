import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6), dpi=250)
ax = fig.add_subplot(projection='3d')

x, y = np.mgrid[0:1:1000j, 0:1:1000j]
z = x * y

surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=1.0,
                        antialiased=False, linewidth=0, edgecolor='none')

# Set view angle (elevation and azimuth)
ax.view_init(elev=15, azim=-15)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.tight_layout()

plt.savefig('figure-xy-surface.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-xy-surface.png', dpi=300, bbox_inches='tight')

plt.close()
print("figure-xy-surface saved successfully")
