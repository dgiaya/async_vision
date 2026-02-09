import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_trajectory(csv_path, camera_name='cam0', output_path=None):
    #read csv
    df = pd.read_csv(csv_path)

    #clean column names
    df.columns = df.columns.str.strip()

    #extract columns
    timestamps = df['#timestamp']
    x = df['p_RS_R_x [m]']  
    y = df['p_RS_R_y [m]']
    z = df['p_RS_R_z [m]']

    #timestamps in seconds from start
    t_start = timestamps.iloc[0]
    t_seconds = (timestamps - t_start) / 1e9 #nanoseconds to seconds

    if output_path is None:
        output_path = Path(csv_path).parent
    else:
        output_path = Path(output_path)

    # X trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(t_seconds, x, '-o', color='blue', markerfacecolor='black', markeredgecolor='black',linewidth=1, markersize=3)
    plt.ylabel('X Position [m]')
    plt.xlabel('Time [s]')
    plt.title(f'{camera_name} Trajectory - X Position')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'{camera_name}_trajectory_x.png', dpi=150)
    plt.show()
    print(f"Saved X trajectory plot to {output_path / f'{camera_name}_trajectory_x.png'}")

    # Y trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(t_seconds, y, '-o', color='green', markerfacecolor='black', markeredgecolor='black',linewidth=1, markersize=3)
    plt.ylabel('Y [m]')
    plt.xlabel('Time [s]')
    plt.title(f'{camera_name} Trajectory - Y axis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'{camera_name}_trajectory_y.png', dpi=150)
    plt.show()
    print(f"Saved: {output_path / f'{camera_name}_trajectory_y.png'}")
    
    # Z trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(t_seconds, z, '-o', color='red', markerfacecolor='black', markeredgecolor='black', linewidth=1, markersize=3)
    plt.ylabel('Z [m]')
    plt.xlabel('Time [s]')
    plt.title(f'{camera_name} Trajectory - Z axis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'{camera_name}_trajectory_z.png', dpi=150)
    plt.show()
    print(f"Saved: {output_path / f'{camera_name}_trajectory_z.png'}")

    # print stats
    print("Trajectory Statistics:")
    print(f"Duration: {t_seconds.iloc[-1]:.2f} seconds")
    print(f"Number of poses: {len(df)}")

# def subsampled_traj(csv_path, camera_name='cam0', output_path=None):

if __name__ == "__main__":
    csv_path = "/home/sid/async_vision/Kalibr/raw_data/decompressed_ros1-poses-cam0-cam0.csv"
    camera_name = 'cam0'

    plot_trajectory(csv_path, camera_name)