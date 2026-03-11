import numpy as np
import matplotlib.pyplot as plt

# Load data
discrete = np.loadtxt('/home/sid/NeuROAM_data/poses_discrete_cam0_main.csv', delimiter=',', skiprows=1)
spline_dense = np.loadtxt('/home/sid/NeuROAM_data/poses_spline_dense_cam0_main.csv', delimiter=',', skiprows=1)

# columns: t, tx, ty, tz, rx, ry, rz
t_disc = discrete[:, 0]
t_spline = spline_dense[:, 0]

labels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
units = ['m', 'm', 'm', 'rad', 'rad', 'rad']

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig.suptitle('Discrete PnP Poses vs Fitted B-Spline', fontsize=14)

for i, ax in enumerate(axes.flat):
    col = i + 1  # skip time column
    
    # spline as smooth line
    ax.plot(t_spline - t_spline[0], spline_dense[:, col], 
            color='steelblue', linewidth=1.0, label='B-spline (dense)', alpha=0.8)
    
    # discrete poses as dots
    ax.scatter(t_disc - t_spline[0], discrete[:, col], 
               color='red', s=8, zorder=5, label='PnP poses (discrete)')
    
    ax.set_ylabel(f'{labels[i]} ({units[i]})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

axes[2, 0].set_xlabel('Time (s)')
axes[2, 1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('/home/sid/async_vision/Kalibr/spline_vs_discrete.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to /home/sid/async_vision/Kalibr/spline_vs_discrete.png")