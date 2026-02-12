import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Set N value
N = 8
eps = 1e-3/N

faces = []
for j in range(2*N):
    for k in range(2*N):
        x = (j-k)/(2*N) + 0.5
        y = (j+k+1)/(2*N) - 0.5
        if (x >= 0) and (x <= 1) and (y >= 0) and (y <= 1):
            pts = []
            for delta in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                xx = x + delta[0]/(2*N)
                yy = y + delta[-1]/(2*N)
                if (xx >= 0-eps) and (xx <= 1+eps) and (yy >= 0-eps) and (yy <= 1+eps):
                    pts.append([xx, yy, xx*yy])
            pts = np.array(pts)
            faces.append(pts)

faces_triang = []
for face in faces:
    if len(face) == 3:
        faces_triang.append(face)
    else:
        x, y = face.mean(axis=0)[0:2]
        if (y >= x - eps) ^ (x+y >= 1-eps):
            face = face[face[:, 0].argsort(), :]
        else:
            face = face[face[:, 1].argsort(), :]
        faces_triang.append(face[0:3, :])
        faces_triang.append(face[1:4, :])

x_plus = []
y_plus = []
xy_plus = []
x_minus = []
y_minus = []
xy_minus = []

for j in range(2*N+1):
    tmp_x = []
    tmp_y = []
    tmp_xy = []
    for k in range(2*N+1):
        x = (j-k)/(2*N) + 0.5
        y = (j+k)/(2*N) - 0.5
        if (x >= 0-eps) and (x <= 1+eps) and (y >= 0-eps) and (y <= 1+eps):
            tmp_x.append(x)
            tmp_y.append(y)
            tmp_xy.append(x*y)
    x_plus.append(tmp_x)
    y_plus.append(tmp_y)
    xy_plus.append(tmp_xy)

for k in range(2*N+1):
    tmp_x = []
    tmp_y = []
    tmp_xy = []
    for j in range(2*N+1):
        x = (j-k)/(2*N) + 0.5
        y = (j+k)/(2*N) - 0.5
        if (x >= 0-eps) and (x <= 1+eps) and (y >= 0-eps) and (y <= 1+eps):
            tmp_x.append(x)
            tmp_y.append(y)
            tmp_xy.append(x*y)
    x_minus.append(tmp_x)
    y_minus.append(tmp_y)
    xy_minus.append(tmp_xy)

# Create figure with triangulated faces
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
faceCollection = Poly3DCollection(faces_triang, shade=False, facecolors='C1', edgecolors='k', alpha=0.5)
ax.add_collection3d(faceCollection)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'N = {N}, {len(faces_triang)} pieces, error = {100/(4*N)**2:.3f}%')
ax.view_init(elev=30, azim=-45)

plt.tight_layout()
plt.savefig('figure-triangulated-pieces.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-triangulated-pieces.png', dpi=300, bbox_inches='tight')
plt.close()

# Create figure with polygons
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
faceCollection = Poly3DCollection(faces, shade=False, facecolors='C1', edgecolors='k', alpha=0.5)
ax.add_collection3d(faceCollection)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'N = {N}, {len(faces)} pieces, error = {100/(4*N)**2:.3f}%')
ax.view_init(elev=30, azim=-45)

plt.tight_layout()
plt.savefig('figure-polygon-pieces.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-polygon-pieces.png', dpi=300, bbox_inches='tight')
plt.close()

# Create figure with grid lines
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
faceCollection = Poly3DCollection(faces, shade=False, facecolors='C1', alpha=0.5)
for n in range(len(x_plus)):
    ax.plot(x_plus[n], y_plus[n], xy_plus[n], 'C2')
for n in range(len(x_minus)):
    ax.plot(x_minus[n], y_minus[n], xy_minus[n], 'C3')
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 0, 0], 'k')
ax.add_collection3d(faceCollection)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'N = {N}, {2*N} + {2*N} regions, error = {100/(4*N)**2:.3f}%')
ax.view_init(elev=30, azim=-45)

plt.tight_layout()
plt.savefig('figure-grid-lines.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-grid-lines.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"figure-triangulated-pieces saved successfully")
print(f"figure-polygon-pieces saved successfully")
print(f"figure-grid-lines saved successfully")
print(f'Relative error: {100/(4*N)**2:.3f}%')
