"""
Extract timestamps from ROS1 bag file and save to CSV
"""

from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores
import csv
import sys
from pathlib import Path

def extract_timestamps_ros1(
    bag_path,
    cam0_topic='/cam_sync/cam0/image_raw',
    cam1_topic='/cam_sync/cam1/image_raw',
    output_csv='raw_ros1_ts.csv'
):
    """
    Extract timestamps from ROS1 bag file using rosbags library.
    
    Args:
        bag_path: Path to ROS1 .bag file
        cam0_topic: Topic name for camera 0
        cam1_topic: Topic name for camera 1
        output_csv: Output CSV filename
    """
    
    print(f"Opening ROS1 bag: {bag_path}")
    
    # Storage for timestamps
    cam0_data = []  # [(frame_idx, timestamp_sec)]
    cam1_data = []
    
    cam0_count = 0
    cam1_count = 0
    total_messages = 0
    
    try:
        # Create typestore for deserialization
        typestore = get_typestore(Stores.ROS1_NOETIC)
        
        with Reader(bag_path) as reader:
            # Show available topics
            print(f"\nTopics in bag:")
            connections = [x for x in reader.connections]
            for connection in connections:
                if 'image' in connection.topic or 'camera' in connection.topic:
                    print(f"  {connection.topic}: {connection.msgtype}")
            
            print(f"\nExtracting timestamps from:")
            print(f"  cam0: {cam0_topic}")
            print(f"  cam1: {cam1_topic}")
            
            # Read messages - get timestamp from bag time and message
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic not in [cam0_topic, cam1_topic]:
                    continue
                
                total_messages += 1
                
                # Deserialize using typestore's deserialize_ros1 method
                # Pass the typename string, not the type class
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                
                # Get timestamp from message header (in seconds)
                # Handle both ROS1 and ROS2 time formats
                try:
                    if hasattr(msg.header.stamp, 'sec'):
                        # ROS1/ROS2 format with sec and nanosec
                        if hasattr(msg.header.stamp, 'nanosec'):
                            timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                        else:
                            # ROS1 format with sec and nsec
                            timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
                    else:
                        # Fallback to bag timestamp
                        timestamp_sec = timestamp * 1e-9
                except AttributeError:
                    # Use bag timestamp as fallback
                    timestamp_sec = timestamp * 1e-9
                
                if connection.topic == cam0_topic:
                    cam0_data.append((cam0_count, timestamp_sec))
                    cam0_count += 1
                elif connection.topic == cam1_topic:
                    cam1_data.append((cam1_count, timestamp_sec))
                    cam1_count += 1
                
                # Progress indicator
                if total_messages % 100 == 0:
                    print(f"Processed {total_messages} messages...", end='\r')
        
        print(f"\nExtracted {len(cam0_data)} cam0 frames, {len(cam1_data)} cam1 frames")
        
    except Exception as e:
        print(f"Error reading bag file: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Save to CSV
    if cam0_data or cam1_data:
        save_timestamps_csv(cam0_data, cam1_data, output_csv)
    
    return cam0_data, cam1_data


def save_timestamps_csv(cam0_data, cam1_data, output_csv):
    """
    Save timestamps to CSV file.
    
    Args:
        cam0_data: List of (frame_idx, timestamp) for cam0
        cam1_data: List of (frame_idx, timestamp) for cam1
        output_csv: Output CSV filename
    """
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'pair_idx',
            'cam0_frame_idx',
            'cam0_timestamp_sec',
            'cam1_frame_idx',
            'cam1_timestamp_sec',
            'time_diff_ms'
        ])

        max_rows = max(len(cam0_data), len(cam1_data))
        
        for i in range(max_rows):
            cam0_idx = cam0_data[i][0] if i < len(cam0_data) else None
            cam0_time = cam0_data[i][1] if i < len(cam0_data) else None
            
            cam1_idx = cam1_data[i][0] if i < len(cam1_data) else None
            cam1_time = cam1_data[i][1] if i < len(cam1_data) else None
            
            # time difference
            if cam0_time is not None and cam1_time is not None:
                time_diff = (cam1_time - cam0_time) * 1000  # Convert to ms
            else:
                time_diff = None
            
            writer.writerow([
                i,
                cam0_idx if cam0_idx is not None else '',
                f'{cam0_time:.9f}' if cam0_time is not None else '',
                cam1_idx if cam1_idx is not None else '',
                f'{cam1_time:.9f}' if cam1_time is not None else '',
                f'{time_diff:.3f}' if time_diff is not None else ''
            ])
    
    print(f"\nTimestamps saved to: {output_csv}")
    
    # print stats
    if cam0_data and cam1_data:
        time_diffs = []
        for i in range(min(len(cam0_data), len(cam1_data))):
            diff = (cam1_data[i][1] - cam0_data[i][1]) * 1000
            time_diffs.append(diff)
        
        if time_diffs:
            avg_diff = sum(time_diffs) / len(time_diffs)
            min_diff = min(time_diffs)
            max_diff = max(time_diffs)
            print(f"\nTimestamp difference statistics:")
            print(f"  Average: {avg_diff:.3f} ms")
            print(f"  Min: {min_diff:.3f} ms")
            print(f"  Max: {max_diff:.3f} ms")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        bag_path = sys.argv[1]
    else:
        bag_path = '/home/sid/NeuROAM_data/merged_payload4b/decompressed_ros1.bag'
    
    # timestamps into csv
    cam0_data, cam1_data = extract_timestamps_ros1(
        bag_path=bag_path,
        cam0_topic='/cam_sync/cam0/image_raw',
        cam1_topic='/cam_sync/cam1/image_raw',
        output_csv='/home/sid/NeuROAM_data/merged_payload4b/raw_data_results/decompressed_ros1_ts.csv'
    )

