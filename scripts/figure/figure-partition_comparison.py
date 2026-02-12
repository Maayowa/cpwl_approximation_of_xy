import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from cpwllib.implementation import N_from_target_error, list_faces_from_N

target_error = 0.001
N = N_from_target_error(target_error)

# Use smaller N for clearer visualization
N = 5
faces_T = list_faces_from_N(N, method='triangles')
faces_P = list_faces_from_N(N, method='polygons')

point = (0.3, 0.4)
is_in_polygon = [Path(f[:, :2]).contains_point(point) for f in faces_P]
selected_polygon = faces_P[np.where(is_in_polygon)[0][0]]
is_triangle_in_polygon = [Path(selected_polygon[:, :2]).contains_point(tuple(f[:, :2].mean(axis=0))) for f in faces_T]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
j = 0
k = 0
ax[0].plot([-1, 2], [j/N+1, j/N-2], 'k')
ax[0].plot([-1, 2], [-1-1+j/N, 2-1+j/N], 'k--')
ax[0].add_collection(PolyCollection([selected_polygon[:, :2]], facecolors='C2', alpha=0.6))
for j in range(2*N):
    ax[0].plot([-1, 2], [j/N+1, j/N-2], 'k')
    ax[0].plot([-1, 2], [-1-1+j/N, 2-1+j/N], 'k--')
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_title('"DC" and "polygon" representation of $g_n$')
ax[0].legend(['Boundary between linear domains of $g_n^+$',
                'Boundary between linear domains of $g_n^-$',
                'A linear domain of $g_n$'])

ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
faceCollection = PolyCollection([face[:, :2] for face in faces_T], facecolors='#FF000000', edgecolors='k')
ax[1].add_collection(PolyCollection([selected_polygon[:, :2]], facecolors='C2', alpha=0.6))
ax[1].add_collection(faceCollection)
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_title('"triangle" representation of $g_n$')
ax[1].legend(['Domain of a pair of co-planar pieces'])

fig.tight_layout()

plt.savefig('figure-partition-comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-partition-comparison.png', dpi=300, bbox_inches='tight')

plt.close()
print("figure-partition-comparison saved successfully")
