import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cpwllib.implementation import N_from_target_error, list_faces_from_N, list_faces_from_N_DC

target_error = 0.001
N = N_from_target_error(target_error)

# Use smaller N for clearer visualization
N = 5
faces_P = list_faces_from_N(N, method='polygons')
faces_plus, faces_minus = list_faces_from_N_DC(N)

fig = plt.figure(figsize=(10, 4), dpi=150)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
faceCollection = Poly3DCollection(faces_P, shade=False, facecolors='C2', edgecolors='k', alpha=0.5)
ax1.add_collection3d(faceCollection)
ax1.set_title('$g_n$')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
faceCollection = Poly3DCollection(faces_plus, shade=False, facecolors='C1', edgecolors='k', alpha=0.5)
ax2.add_collection3d(faceCollection)
faceCollection = Poly3DCollection(faces_minus, shade=False, facecolors='C0', edgecolors='k', alpha=0.5)
ax2.add_collection3d(faceCollection)
ax2.set_title('$g_n^+$ and $g_n^-$')
ax2.legend(['$g_n^+$', '$g_n^-$'])

for ax in [ax1, ax2]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=30, azim=-45)

fig.tight_layout()

plt.savefig('figure-dc-comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-dc-comparison.png', dpi=300, bbox_inches='tight')

plt.close()
print("figure-dc-comparison saved successfully")
