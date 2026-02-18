import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import csv
from pathlib import Path
import sys

def extract_timestamps(
        bag_path,
        cam0_topic='/cam_sync/cam0/image_raw',
        cam1_topic='/cam_sync/cam1/image_raw',
        output_csv='timestamps_skipped.csv'
):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=''
    )

    converter_options = rosbag2_py.ConverterOptions('', '')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    cam0_msgtype = None
    cam1_msgtype = None

    for topic_metadata in topic_types:
        if topic_metadata.name == cam0_topic:
            cam0_msgtype = topic_metadata.type
        elif topic_metadata.name == cam1_topic:
            cam1_msgtype = topic_metadata.type

    print(f"Found topics:")
    print(f"  {cam0_topic} ({cam0_msgtype})")
    print(f"  {cam1_topic} ({cam1_msgtype})")

    #message class
    try:
        Cam0MsgType = get_message(cam0_msgtype)
        Cam1MsgType = get_message(cam1_msgtype)
    except Exception as e:
        print(f"Error loading message types: {e}")
        # Fallback
        Cam0MsgType = get_message('sensor_msgs/msg/Image')
        Cam1MsgType = get_message('sensor_msgs/msg/Image')

    #timestamp store
    cam0_data = []  # (frame_idx, timestamp)
    cam1_data = []

    cam0_count = 0
    cam1_count = 0
    total_messages = 0

    while reader.has_next():
        topic, data, bag_timestamp = reader.read_next()
        total_messages += 1
        
        if topic == cam0_topic:
            msg = deserialize_message(data, Cam0MsgType)
            # Get timestamp from header
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cam0_data.append((cam0_count, timestamp))
            cam0_count += 1
            
        elif topic == cam1_topic:
            msg = deserialize_message(data, Cam1MsgType)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cam1_data.append((cam1_count, timestamp))
            cam1_count += 1

    print(f"\nExtracted {len(cam0_data)} cam0 frames, {len(cam1_data)} cam1 frames")
    
    del reader

    #save csv
    save_timestamps_csv(cam0_data, cam1_data, output_csv)

    return cam0_data, cam1_data

def save_timestamps_csv(cam0_data, cam1_data, output_csv):
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

        # Write rows
        max_rows = max(len(cam0_data), len(cam1_data))
        
        for i in range(max_rows):
            cam0_idx = cam0_data[i][0] if i < len(cam0_data) else None
            cam0_time = cam0_data[i][1] if i < len(cam0_data) else None
            
            cam1_idx = cam1_data[i][0] if i < len(cam1_data) else None
            cam1_time = cam1_data[i][1] if i < len(cam1_data) else None
            
            # Calculate time diff if both exist
            if cam0_time is not None and cam1_time is not None:
                time_diff = (cam1_time - cam0_time) * 1000
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

#############

if __name__ == "__main__":
    if len(sys.argv) > 1:
        bag_path = sys.argv[1]
    else:
        bag_path = '/home/sid/ROAM_bag/calib_unsync_100ms'

    # Extract timestamps
    cam0_data, cam1_data = extract_timestamps(
        bag_path=bag_path,
        cam0_topic='/cam_sync/cam0/image_raw',
        cam1_topic='/cam_sync/cam1/image_raw',
        output_csv='timestamp_100ms.csv'
    )