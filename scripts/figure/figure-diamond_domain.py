import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure(figsize=(10, 5), dpi=150, constrained_layout=True)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

ax = []
ax.append(fig.add_subplot(gs[0]))
ax.append(fig.add_subplot(gs[1]))

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

# Parameters
x0, y0 = 0, 0
delta = 1

# Define diamond corners
diamond = np.array([
    [x0,         y0 + delta],
    [x0 + delta, y0        ],
    [x0,         y0 - delta],
    [x0 - delta, y0        ],
    [x0,         y0 + delta]
])

# Plot diamond
ax[0].plot(diamond[:, 0], diamond[:, 1], 'k-', lw=2, label='Diamond Square')

# Plot centroid
ax[0].text(x0+0.05, y0+0.05, f'$(x_0,y_0)$')

# Plot axes
for k in [-1, 0, 1]:
    ax[0].axhline(y=y0+k*delta, color='gray', linestyle='--', linewidth=1)
    ax[0].axvline(x=x0+k*delta, color='gray', linestyle='--', linewidth=1)

# Define lines (edges of the diamond)
lines = [
    ((x0 - delta, y0), (x0, y0 + delta)),
    ((x0, y0 + delta), (x0 + delta, y0)),
    ((x0 + delta, y0), (x0, y0 - delta)),
    ((x0, y0 - delta), (x0 - delta, y0))
]

offset = 0.3  # outward distance from center for label placement

for (x1, y1), (x2, y2) in lines:
    # Midpoint
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    # Direction vector of the edge
    dx = x2 - x1
    dy = y2 - y1

    # Normal vector pointing outward from (x0, y0)
    nx = y_mid - y0
    ny = -(x_mid - x0)
    norm = np.hypot(nx, ny)
    nx /= norm
    ny /= norm

    # Offset position away from center
    x_text = x_mid
    y_text = y_mid + 0.05

    # Compute slope and angle
    m = (y2 - y1) / (x2 - x1)
    angle = np.degrees(np.arctan2(dy, dx))

    # Ensure text is upright
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # Place label
    ax[0].text(
        x_text, y_text,
        rf"$y = y_0 {'-' if m<0 else '+'}x {'+' if m<0 else '-'}x_0 {'+' if (x_mid-x0)*m<0 else '-'}\delta$",
        fontsize=9,
        rotation=angle,
        rotation_mode='anchor',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='none', alpha=0.7)
    )

# Symbolic tick labels
xticks = [x0 - delta, x0, x0 + delta]
yticks = [y0 - delta, y0, y0 + delta]
ax[0].set_xticks(xticks)
ax[0].set_yticks(yticks)
ax[0].set_xticklabels([r'$x_0 - \delta$', r'$x_0$', r'$x_0 + \delta$'])
ax[0].set_yticklabels([r'$y_0 - \delta$', r'$y_0$', r'$y_0 + \delta$'])

# Final formatting
ax[0].set_aspect('equal')
ratio_lim = 1.0
ax[0].set_xlim(x0 - ratio_lim * delta, x0 + ratio_lim * delta)
ax[0].set_ylim(y0 - ratio_lim * delta, y0 + ratio_lim * delta)
ax[0].set_title('$R_0$ domain')

# Alternative plot for |E|

# Create a square grid
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
x, y = np.meshgrid(x, y)

z = x * y

mask_idx = (x + y <= 1) & (x + y >= -1) & (y - x <= 1) & (y - x >= -1)

z[~mask_idx] = np.nan

# Plot
c = ax[1].pcolormesh(x, y, abs(z), cmap='viridis', shading='auto')
ax[1].set_aspect('equal')

# Add colorbar
cb_ax = fig.add_subplot(gs[2])
cb = fig.colorbar(c, cax=cb_ax)

# Define diamond corners
diamond = np.array([
    [x0,         y0 + delta],
    [x0 + delta, y0        ],
    [x0,         y0 - delta],
    [x0 - delta, y0        ],
    [x0,         y0 + delta]
])

# Plot diamond
ax[1].plot(diamond[:, 0], diamond[:, 1], 0, 'black', linewidth=0.3)

# Custom x and y ticks
ax[1].set_xticks([x0 - delta, x0, x0 + delta])
ax[1].set_xticklabels([r'$x_0 - \delta$', r'$x_0$', r'$x_0 + \delta$'])

ax[1].set_yticks([y0 - delta, y0, y0 + delta])
ax[1].set_yticklabels([r'$y_0 - \delta$', r'$y_0$', r'$y_0 + \delta$'])

ax[1].set_xlabel('x')

# Set abstract tick labels based on \delta
cb.set_ticks([0, 0.25])
cb.set_ticklabels([r'$0$', r'$\delta^2/4$'])

ax[1].set_title(r'Value of $|E(x,y)|$')

plt.savefig('figure-diamond-domain.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-diamond-domain.png', dpi=300, bbox_inches='tight')

plt.close()
print("figure-diamond-domain saved successfully")
