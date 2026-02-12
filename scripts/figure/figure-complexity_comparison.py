import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter


# Custom formatter to remove trailing ".0"
class NoDecimalScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%d"  # integer formatting only


# n from 1 to 32
n = 2**np.arange(6)

# Functions from the table (linear constraints)
triangle = 16*n**2 + 4
polygon = 10*n*(n+1) + 4
dc = 14*n + 9
triangle_log = 16*n**2 + 2*np.ceil(2*np.log2(2*n)) + 4
polygon_log = 10*n*(n+1) + 2*np.ceil(np.log2(2*n*(n+1))) + 4
dc_log = 14*n + 4*np.ceil(np.log2(2*n)) + 9

# Continuous variables
triangle_cont = 12*n**2 + 1
polygon_cont = 6*n*(n+1) + 1
dc_cont = 12*n + 3
triangle_log_cont = 16*n**2 + 1
polygon_log_cont = 8*n*(n+1) + 1
dc_log_cont = 16*n + 3

# Binary variables
triangle_bin = 4*n**2
polygon_bin = 2*n*(n+1)
dc_bin = 4*n
triangle_log_bin = np.ceil(2*np.log2(2*n))
polygon_log_bin = np.ceil(np.log2(2*n*(n+1)))
dc_log_bin = 2*np.ceil(np.log2(2*n))

# Approximation error
error = 1/(16*n**2)

# Create the plot
fig, ax1 = plt.subplots(1, 3, figsize=(10, 4), dpi=150)
plt.subplots_adjust(wspace=0.0)

# Primary y-axis: linear constraints
l1, = ax1[0].plot(n, triangle, '--', color='C0', label='Triangle')
l2, = ax1[0].plot(n, polygon, '--', color='C1', label='Polygon')
l3, = ax1[0].plot(n, dc, '--', color='C2', label='DC')
l4, = ax1[0].plot(n, triangle_log, '-', color='C0', label='Triangle (LogEnc)')
l5, = ax1[0].plot(n, polygon_log, '-', color='C1', label='Polygon (LogEnc)')
l6, = ax1[0].plot(n, dc_log, '-', color='C2', label='DC (LogEnc)')

ax1[0].set_xscale('log', base=2)
ax1[0].set_yscale('log')
ax1[0].set_xlabel('n')
ax1[0].set_ylabel('Number of constraints, variables')
ax1[0].set_ylim([1, 10**5])

ax1[0].set_title('Linear constraints')

# Apply the integer-only formatter
formatter = NoDecimalScalarFormatter()
formatter.set_scientific(False)
ax1[0].xaxis.set_major_formatter(formatter)
ax1[0].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))

# Secondary y-axis: approximation error
ax2 = ax1[0].twinx()
l7, = ax2.plot(n, error, 'C3', label='Error')
ax2.set_yscale('log')
ax2.set_ylim([10**(-5), 10**(0)])
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
ax2.set_yticklabels([])

ax1[0].grid(True, which="both", ls="--", alpha=0.5)

# Second figure - Binary variables
ax1[1].plot(n, triangle_bin, '--', color='C0', label='Triangle')
ax1[1].plot(n, polygon_bin, '--', color='C1', label='Polygon')
ax1[1].plot(n, dc_bin, '--', color='C2', label='DC')
ax1[1].plot(n, triangle_log_bin, '-', color='C0', label='Triangle (LogEnc)')
ax1[1].plot(n, polygon_log_bin, '-', color='C1', label='Polygon (LogEnc)')
ax1[1].plot(n, dc_log_bin, '-', color='C2', label='DC (LogEnc)')

ax1[1].set_xscale('log', base=2)
ax1[1].set_yscale('log')
ax1[1].set_xlabel('n')
ax1[1].set_ylim([1, 10**5])

ax1[1].set_title('Binary variables')

# Apply the integer-only formatter
formatter = NoDecimalScalarFormatter()
formatter.set_scientific(False)
ax1[1].xaxis.set_major_formatter(formatter)
ax1[1].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
ax1[1].set_yticklabels([])

# Secondary y-axis: approximation error
ax2 = ax1[1].twinx()
ax2.plot(n, error, 'C3', label='Error')
ax2.set_yscale('log')
ax2.set_ylim([10**(-5), 10**(0)])
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
ax2.set_yticklabels([])

ax1[1].grid(True, which="both", ls="--", alpha=0.5)

# Third figure - Continuous variables
ax1[2].plot(n, triangle_cont, '--', color='C0', label='Triangle')
ax1[2].plot(n, polygon_cont, '--', color='C1', label='Polygon')
ax1[2].plot(n, dc_cont, '--', color='C2', label='DC')
ax1[2].plot(n, triangle_log_cont, '-', color='C0', label='Triangle (LogEnc)')
ax1[2].plot(n, polygon_log_cont, '-', color='C1', label='Polygon (LogEnc)')
ax1[2].plot(n, dc_log_cont, '-', color='C2', label='DC (LogEnc)')

ax1[2].set_xscale('log', base=2)
ax1[2].set_yscale('log')
ax1[2].set_xlabel('n')
ax1[2].set_ylim([1, 10**5])
ax1[2].set_yticklabels([])

ax1[2].set_title('Continuous variables')

# Apply the integer-only formatter
formatter = NoDecimalScalarFormatter()
formatter.set_scientific(False)
ax1[2].xaxis.set_major_formatter(formatter)
ax1[2].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))

# Secondary y-axis: approximation error
ax2 = ax1[2].twinx()
ax2.plot(n, error, 'C3', label='Error')
ax2.set_yscale('log')
ax2.set_ylabel('Maximum approximation error')
ax2.set_ylim([10**(-5), 10**(0)])
ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))

ax1[2].grid(True, which="both", ls="--", alpha=0.5)

# Make legend belong to the figure
fig.legend(
    handles=[l1, l2, l3, l4, l5, l6, l7],
    loc='lower center',
    bbox_to_anchor=(0.5, +0.05),
    fontsize=9,
    ncol=7
)

plt.subplots_adjust(bottom=0.28)

plt.savefig('figure-complexity-comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure-complexity-comparison.png', dpi=300, bbox_inches='tight')

plt.close()
print("figure-complexity-comparison saved successfully")
