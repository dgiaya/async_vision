import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(json_path):
    """Load calibration results from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_detection_analysis(results, save_path='output/detection_analysis.png'):
    poses = results['poses']
    tags_per_frame = [p['num_tags_detected'] for p in poses]
    timestamps = [p['timestamp'] for p in poses]
    
    # Sort by timestamp for temporal plot
    sorted_indices = np.argsort(timestamps)
    timestamps_sorted = np.array(timestamps)[sorted_indices]
    tags_temporal = np.array([poses[i]['num_tags_detected'] for i in sorted_indices])
    time_relative = timestamps_sorted - timestamps_sorted[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Detection Robustness
    axes[0].hist(tags_per_frame, 
                bins=range(10, max(tags_per_frame)+2),
                edgecolor='black', alpha=0.7, color='mediumseagreen')
    axes[0].axvline(np.mean(tags_per_frame), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(tags_per_frame):.1f}')
    axes[0].axvline(10, color='orange', linestyle='--', linewidth=2,
                   label='Min threshold (10)')
    axes[0].set_xlabel('Number of Tags Detected', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_xlim(left=10)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].set_title('Detection Robustness', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Detection Consistency Over Time
    axes[1].plot(time_relative, tags_temporal, linewidth=1, color='green', alpha=0.7)
    axes[1].axhline(np.mean(tags_temporal), color='darkgreen', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(tags_temporal):.1f}')
    axes[1].axhline(10, color='orange', linestyle='--', linewidth=1.5,
                   label='Min threshold (10)')
    axes[1].fill_between(time_relative, 0, tags_temporal, alpha=0.2, color='green')
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_ylabel('Tags Detected', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].set_title('Detection Consistency Over Time', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()

def plot_reprojection_error_distribution(results, save_path='output/reproj_error_dist.png'):
    """Reprojection error histogram"""
    poses = results['poses']
    errors = [p['reprojection_error'] for p in poses]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(errors):.3f} px')
    ax.set_xlabel('Reprojection Error (pixels)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title('Distribution of Reprojection Errors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # sub-pixel accuracy stat
    subpixel_pct = 100 * sum(e < 1.0 for e in errors) / len(errors)
    textstr = f'Total frames: {len(errors)}\n'
    textstr += f'Sub-pixel (<1.0 px): {subpixel_pct:.1f}%\n'
    textstr += f'Std: {np.std(errors):.3f} px'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()

def plot_camera_trajectory_and_error(results, save_path='output/trajectory_and_error.png'):
    poses = results['poses']
    
    timestamps = [p['timestamp'] for p in poses]
    sorted_indices = np.argsort(timestamps)
    timestamps_sorted = np.array(timestamps)[sorted_indices]
    
    # Calculate frame indices based on 20Hz sampling
    time_start = timestamps_sorted[0]
    frame_indices = []
    
    for ts in timestamps_sorted:
        frame_num = int(round((ts - time_start) * 20))
        frame_indices.append(frame_num)
    
    # Get reprojection errors in sorted order
    reproj_errors = np.array([poses[i]['reprojection_error'] for i in sorted_indices])
    
    # Extract camera positions
    positions = []
    orientations = []
    for idx in sorted_indices:
        T_inv = np.array(poses[idx]['T_inv'])
        positions.append(T_inv[:3, 3])
        orientations.append(T_inv[:3, :3])
    positions = np.array(positions)
    
    # Calculate statistics
    mean_error = np.mean(reproj_errors)
    std_error = np.std(reproj_errors)
    traj_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    
    # Panel 1: 3D Trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot solid blue trajectory line
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
        color='steelblue', linewidth=2, alpha=0.8, label='Camera Path')
    
    # Black dots at frame positions
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
           c='black', s=8, alpha=0.6, zorder=5)
    
    # Start and End markers
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, marker='D', label='Start', 
               edgecolors='black', linewidths=2, zorder=10)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=100, marker='s', label='End', 
               edgecolors='black', linewidths=2, zorder=10)
    
    # Camera orientations every 15 frames
    step = 15
    axis_length = 0.05
    
    for i in range(0, len(positions), step):
        pos = positions[i]
        R_mat = orientations[i]
        
        # X-axis (red), Z-axis (blue)
        x_end = pos + axis_length * R_mat[:, 0]
        z_end = pos + axis_length * R_mat[:, 2]
        
        ax1.plot([pos[0], x_end[0]], [pos[1], x_end[1]], [pos[2], x_end[2]], 
                'r-', linewidth=1.5, alpha=0.4)
        ax1.plot([pos[0], z_end[0]], [pos[1], z_end[1]], [pos[2], z_end[2]], 
                'b-', linewidth=1.5, alpha=0.4)
    
    ax1.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (meters)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Camera 0 Trajectory\n({len(positions)} poses, {traj_length:.1f}m path)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Reprojection Error Stem Plot
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Stem plot using actual frame indices
    markerline, stemlines, baseline = ax2.stem(frame_indices, reproj_errors,
                                               linefmt='b-', markerfmt='bx',
                                               basefmt='k-')
    markerline.set_markersize(4)
    markerline.set_markeredgewidth(1)
    stemlines.set_linewidth(1)
    stemlines.set_alpha(0.6)
    
    # Mean line
    ax2.axhline(y=mean_error, color='r', linestyle='--', linewidth=2,
               label=f'Mean: {mean_error:.3f} px', zorder=5)
    
    ax2.set_xlabel('Frame Index (from bag file)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Reprojection Error (pixels)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Reprojection Error per Frame\n({len(poses)} successful frames shown)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    
    # Statistics box
    textstr = f'Statistics:\n'
    textstr += f'Mean:  {mean_error:.3f} px\n'
    textstr += f'Std:   {std_error:.3f} px\n'
    textstr += f'Min:   {np.min(reproj_errors):.3f} px\n'
    textstr += f'Max:   {np.max(reproj_errors):.3f} px\n'
    textstr += f'<1px:  {sum(reproj_errors < 1.0)}/{len(reproj_errors)} '
    textstr += f'({100*sum(reproj_errors < 1.0)/len(reproj_errors):.1f}%)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()

def plot_3d_trajectory_interactive(results, save_html='output/trajectory_3d.html'):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed. Install with: pip install plotly")
        return
    
    poses = results['poses']
    
    # Extract positions sorted by timestamp
    timestamps = [p['timestamp'] for p in poses]
    sorted_indices = np.argsort(timestamps)
    
    positions = []
    orientations = []
    
    for idx in sorted_indices:
        T_inv = np.array(poses[idx]['T_inv'])
        positions.append(T_inv[:3, 3])
        orientations.append(T_inv[:3, :3])
    
    positions = np.array(positions)
    
    # Calculate path length
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    bbox = positions.max(axis=0) - positions.min(axis=0)
    
    # Create figure
    fig = go.Figure()
    
    # Trajectory path
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=np.arange(len(positions)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Frame", x=1.02)
        ),
        line=dict(color='lightblue', width=2),
        name='Camera Path',
        hovertemplate='<b>Frame %{marker.color}</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<br>Z: %{z:.3f}m<extra></extra>'
    ))
    
    # Start marker
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='diamond'),
        name='Start',
        hovertemplate='<b>Start</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<br>Z: %{z:.3f}m<extra></extra>'
    ))
    
    # End marker
    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='square'),
        name='End',
        hovertemplate='<b>End</b><br>X: %{x:.3f}m<br>Y: %{y:.3f}m<br>Z: %{z:.3f}m<extra></extra>'
    ))
    
    # Camera orientations every 15 frames
    step = max(1, len(positions) // 15)
    axis_length = 0.05
    
    for i in range(0, len(positions), step):
        pos = positions[i]
        R_mat = orientations[i]
        
        # X-axis (red)
        x_end = pos + axis_length * R_mat[:, 0]
        fig.add_trace(go.Scatter3d(
            x=[pos[0], x_end[0]], y=[pos[1], x_end[1]], z=[pos[2], x_end[2]],
            mode='lines', line=dict(color='red', width=2),
            showlegend=False, hoverinfo='skip'
        ))
        
        # Z-axis (blue)
        z_end = pos + axis_length * R_mat[:, 2]
        fig.add_trace(go.Scatter3d(
            x=[pos[0], z_end[0]], y=[pos[1], z_end[1]], z=[pos[2], z_end[2]],
            mode='lines', line=dict(color='blue', width=2),
            showlegend=False, hoverinfo='skip'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'Camera 0 Trajectory (PnP Initialization)<br>' +
                 f'<sub>{len(positions)} poses | {path_length:.1f}m path | Workspace: {bbox[0]:.2f}x{bbox[1]:.2f}x{bbox[2]:.2f}m</sub>',
            x=0.5, xanchor='center', font=dict(size=15)
        ),
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1000,
        height=800
    )
    
    Path(save_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_html)
    print(f"Saved interactive 3D: {save_html}")
    print(f"  Path length: {path_length:.2f} m, Workspace: {np.prod(bbox):.3f} m^3")


if __name__ == "__main__":
    JSON_PATH = "output/cam0_trajectory_apriltag.json"
    results = load_results(JSON_PATH)
    
    print(f"Loaded {len(results['poses'])} poses from {JSON_PATH}\n")
    
    # Generate plots
    print("Generating visualizations...\n")
    
    # Detection analysis
    plot_detection_analysis(results, save_path='output/detection_analysis.png')
    
    # Reprojection error distribution
    plot_reprojection_error_distribution(results, save_path='output/reproj_error_dist.png')
    
    # Camera trajectory + reprojection error stem plot
    plot_camera_trajectory_and_error(results, save_path='output/trajectory_and_error.png')
    
    # Interactive 3D trajectory
    plot_3d_trajectory_interactive(results, save_html='output/trajectory_3d.html')
