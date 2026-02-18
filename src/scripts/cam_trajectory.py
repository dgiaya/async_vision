import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SE3BSplineTrajectory:
    def __init__(self, timestamps, poses, smoothing=0):
        if len(timestamps) != len(poses):
            raise ValueError("timestamps and poses must have same length")
        
        if len(timestamps) < 4:
            raise ValueError("At least 4 poses required for cubic B-spline")
        
        self.timestamps = timestamps
        self.t_min = timestamps[0]
        self.t_max = timestamps[-1]
        self.poses = poses
        self.n_samples = len(timestamps)

        #translation and rotation components
        self.translations = poses[:, :3] #shape (N, 3)
        self.quaternions = poses[:, 3:] #shape (N, 4) [qx, qy, qz, qw]

        # B-spline to translation
        # splprep returns (tck, u) where:
        #   tck = (knot_vector, coefficients, degree) -- (t, c, k)
        #   u = parameter values (we use timestamps)
        self.tck_position, self.u = splprep(
            self.translations.T,  # Transpose, shape (3, N) for x, y, z vector 
            u=timestamps,       # timestamps as parameter values
            k=3,                # Cubic spline
            s=smoothing         # Smoothing factor
        )

        # Slerp interpolator for rotations (quaternion manifold)
        rotations = Rotation.from_quat(self.quaternions)  
        self.rotation_slerp = Slerp(timestamps, rotations)

        print(f"SE3BSplineTrajectory initialized with {len(timestamps)} poses")
        print(f"  Time range: {self.t_min:.3f} to {self.t_max:.3f} seconds")
        print(f"  Duration: {self.t_max - self.t_min:.3f} seconds")

    def query_pose(self, t):
        """
        Query pose at arbitrary timestamp

        parameters:
        t: float

        returns:
        pose: np.ndarray, shape(7,) [x, y, z, qx, qy, qz, qw]
        """
        if t < self.t_min or t > self.t_max:
            raise ValueError(
                f"Timestamp {t:.3f} outside trajectory range"
                f"[{self.t_min:.3f}, {self.t_max:.3f}]"
            )
        
        # interpolate position using B spline
        position = np.array(splev(t, self.tck_position)) # shape (3,)

        # interpolate rotation using Slerp
        rotation = self.rotation_slerp(t)
        quaternion = rotation.as_quat()  #shape (4,) as [qx,qy,qz,qw]

        #full pose
        pose = np.concatenate([position, quaternion])

        return pose
    
    def query_poses(self, timestamps):
        '''
        parameters:
        timestamps: np.ndarray or list  --> array of timestamps

        returns:
        poses: np.ndarray, shape(M, 7)
        '''
        timestamps = np.asarray(timestamps)
        poses = np.array([self.query_pose(t) for t in timestamps])
        return poses
    
    def traj_error(self):
        '''
        how well the B-spline fits the original discrete poses
        stats to check the fitting is valid

        returns:
        error stats: dict
        '''
        # query the poses from the spline trajectory
        splined_poses = self.query_poses(self.timestamps)

        #error
        position_errors = np.linalg.norm(
            splined_poses[:, :3] - self.translations, axis=1
        )

        rotation_errors = []
        for i in range(len(self.timestamps)):
            R_original = Rotation.from_quat(self.quaternions[i])
            R_fitted = Rotation.from_quat(splined_poses[i, 3:])

            #Rotation error as angle in degrees
            dR = R_fitted * R_original.inv()
            angle_error = np.degrees(dR.magnitude())
            rotation_errors.append(angle_error)

        rotation_errors = np.array(rotation_errors)

        return {
            'position_rmse': np.sqrt(np.mean(position_errors**2)),
            'position_max': np.max(position_errors),
            'rotation_rmse': np.sqrt(np.mean(rotation_errors**2)),
            'rotation_max': np.max(rotation_errors),
            'position_errors': position_errors,
            'rotation_errors': rotation_errors
        }


def load_kalibr_poses(csv_path):
    '''
    returns:
    timestamps: np.ndarray, shape(N,)
    poses: np.ndarray, shape(N, 7)
    '''
    df = pd.read_csv(csv_path)

    #verify columns of csv file
    col_check = ['#timestamp', ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]', 
                 ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']
    
    missing = set(col_check) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # extract data
    timestamps = df['#timestamp'].values
    timestamps = timestamps / 1e9  # nanoseconds to seconds
    timestamps = timestamps - timestamps[0]  # Normalize to start at 0

    poses = df[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]', 
                ' q_RS_x []', ' q_RS_y []', ' q_RS_z []', ' q_RS_w []']].values

    print(f"Loaded {len(timestamps)} poses from: {csv_path}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")

    return timestamps, poses


def camera_trajectory(trajectory, frustum_stride=5, frustum_size=0.04, frustum_color='black'):
    """
    Visualize camera trajectory with default X, Y, Z axes.
    
    Parameters:
    -----------
    trajectory : SE3BSplineTrajectory
    frustum_stride : int - show every Nth frustum
    frustum_size : float - size of camera frustums
    frustum_color : str or tuple - color of frustum
    """
    # Sample trajectory densely for smooth curve
    timestamps_dense = np.arange(trajectory.t_min, trajectory.t_max, 0.01)
    poses_dense = trajectory.query_poses(timestamps_dense)
    positions = poses_dense[:, :3]
    
    # Create plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trajectory line (continuous)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2.5, label='Camera Trajectory', alpha=0.8)
    
    # Discrete sample points
    ax.scatter(trajectory.translations[:, 0],
               trajectory.translations[:, 1],
               trajectory.translations[:, 2],
               c='red', s=40, marker='o', 
               label=f'Discrete Samples (n={trajectory.n_samples})',
               edgecolors='darkred', linewidths=1.2, zorder=5, alpha=0.8)
    
    # Camera frustums at selected poses
    print(f"Plotting {trajectory.n_samples // frustum_stride} camera frustums...")
    for i in range(0, trajectory.n_samples, frustum_stride):
        pose = trajectory.poses[i]
        position = pose[:3]
        quaternion = pose[3:]

        R = Rotation.from_quat(quaternion).as_matrix()

        frustum_points_local = np.array([
            [0, 0, 0],
            [-frustum_size, -frustum_size, 2*frustum_size],
            [frustum_size, -frustum_size, 2*frustum_size],
            [frustum_size, frustum_size, 2*frustum_size],
            [-frustum_size, frustum_size, 2*frustum_size],
        ])

        # Transform to world frame
        frustum_points_world = (R @ frustum_points_local.T).T + position

        # Draw frustum edges
        for j in range(1, 5):
            ax.plot([frustum_points_world[0, 0], frustum_points_world[j, 0]],
                   [frustum_points_world[0, 1], frustum_points_world[j, 1]],
                   [frustum_points_world[0, 2], frustum_points_world[j, 2]],
                   color=frustum_color, linewidth=0.8, alpha=0.6)
        
        for j in range(1, 5):
            next_j = j + 1 if j < 4 else 1
            ax.plot([frustum_points_world[j, 0], frustum_points_world[next_j, 0]],
                   [frustum_points_world[j, 1], frustum_points_world[next_j, 1]],
                   [frustum_points_world[j, 2], frustum_points_world[next_j, 2]],
                   color=frustum_color, linewidth=0.8, alpha=0.6)

    # Start marker (green star)
    ax.scatter(trajectory.translations[0, 0],
               trajectory.translations[0, 1],
               trajectory.translations[0, 2],
               c='green', s=200, marker='*', 
               label='Start', edgecolors='darkgreen', 
               linewidths=2, zorder=6)
    
    # End marker (purple X)
    ax.scatter(trajectory.translations[-1, 0],
               trajectory.translations[-1, 1],
               trajectory.translations[-1, 2],
               c='purple', s=200, marker='X', 
               label='End', edgecolors='indigo',
               linewidths=2, zorder=6)
    
    # Labels
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Camera Trajectory - 3D View', 
                 fontsize=14, fontweight='bold', pad=20)

    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Statistics
    stats_text = f"Trajectory Statistics:\n"
    stats_text += f"Samples: {trajectory.n_samples}\n"
    stats_text += f"Duration: {trajectory.t_max - trajectory.t_min:.2f} s"
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    print(f" Camera Trajectory visualization created")
    
    return fig, ax
### main block

if __name__ == "__main__":
    csv_path = '/home/sid/async_vision/Kalibr/raw_data/decompressed_ros1-poses-cam0-cam1.csv'
    timestamps, poses = load_kalibr_poses(csv_path)
    
    # create trajectory
    trajectory = SE3BSplineTrajectory(timestamps, poses)
    
    # verify trajectory quality
    errors = trajectory.traj_error()
    print("Camera Trajectory Quality Statistics:")
    print(f"  Position RMSE: {errors['position_rmse']*1000:.4f} mm")
    print(f"  Position Max:  {errors['position_max']*1000:.4f} mm")
    print(f"  Rotation RMSE: {errors['rotation_rmse']:.6f}°")
    print(f"  Rotation Max:  {errors['rotation_max']:.6f}°")
    
    # visualize 3d trajectory
    print("Creating continuous camera trajectory")
    fig, ax = camera_trajectory(
        trajectory,
        frustum_stride=1,    #how frequent to plot camera boxes
        frustum_size=0.05,      #camera box size
        frustum_color='blue',   
    )
    
    # save figure
    fig.savefig('cam1_traj_96poses', dpi=300, bbox_inches='tight')
