import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

def analyze_timestamps(
        bag_path,
        cam0_topic='/cam_sync/cam0/image_raw/compressed',
        cam1_topic='/cam_sync/cam1/image_raw/compressed',
        output_csv='timestamp_analysis.csv',
        output_plot='timestamp_visualization.png'
):
    cam0_times = []
    cam1_times = []

    with open(bag_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=[cam0_topic, cam1_topic]):
            try:
                timestamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9

                if channel.topic == cam0_topic:
                    cam0_times.append(timestamp)
                elif channel.topic == cam1_topic:
                    cam1_times.append(timestamp)

            except AttributeError:
                ## log time
                timestamp = message.log_time * 1e-9
                
                if channel.topic == cam0_topic:
                    cam0_times.append(timestamp)
                elif channel.topic == cam1_topic:
                    cam1_times.append(timestamp)

    cam0_times = np.array(cam0_times)
    cam1_times = np.array(cam1_times)

    print(f"Collected {len(cam0_times)} cam0 frames")
    print(f"Collected {len(cam1_times)} cam1 frames\n")

    if len(cam0_times) == 0 or len(cam1_times) == 0:
        print("ERROR: No frames found! Check topic names.")
        return None
    
    #stats
    results = compute_sync_statistics(cam0_times, cam1_times)

    #save csv
    save_to_csv(cam0_times, cam1_times, results, output_csv)

    #print summary
    print_summary(results)

    return results

def compute_sync_statistics(cam0_times, cam1_times):
    results = {}

    #duration
    results['duration'] = cam0_times[-1] - cam0_times[0]

    #frame rates
    cam0_intervals = np.diff(cam0_times) * 1000   # in ms
    cam1_intervals = np.diff(cam1_times) * 1000

    results['cam0_rate'] = 1000 / np.mean(cam0_intervals)  # Hz
    results['cam1_rate'] = 1000 / np.mean(cam1_intervals)
    results['cam0_interval_mean'] = np.mean(cam0_intervals)
    results['cam1_interval_mean'] = np.mean(cam1_intervals)
    results['cam0_interval_std'] = np.std(cam0_intervals)
    results['cam1_interval_std'] = np.std(cam1_intervals)
    results['cam0_intervals'] = cam0_intervals
    results['cam1_intervals'] = cam1_intervals

    cam1_times_aligned = cam1_times
    
    # temporal correspondence (aligned)
    time_diffs_aligned = []
    matched_pairs_aligned = []
    
    for i in range(min(len(cam0_times), len(cam1_times_aligned))):
        diff = (cam1_times_aligned[i] - cam0_times[i]) * 1000  # ms
        time_diffs_aligned.append(diff)
        matched_pairs_aligned.append((i, i, diff))  # Store actual indices: (cam0_idx, cam1_idx, diff)

    time_diffs_aligned = np.array(time_diffs_aligned)

    results['index_diff_mean'] = np.mean(np.abs(time_diffs_aligned))
    results['index_diff_std'] = np.std(time_diffs_aligned)
    results['index_diff_min'] = np.min(np.abs(time_diffs_aligned))
    results['index_diff_max'] = np.max(np.abs(time_diffs_aligned))
    results['index_diff_median'] = np.median(np.abs(time_diffs_aligned))

    results['time_diffs'] = time_diffs_aligned
    results['matched_pairs'] = matched_pairs_aligned
    results['cam0_times'] = cam0_times
    results['cam1_times'] = cam1_times
    results['cam1_times_aligned'] = cam1_times_aligned

    return results

def save_to_csv(cam0_times, cam1_times, results, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # CORRECTED header - now matches data rows
        writer.writerow([
            'frame_idx',
            'cam0_idx',
            'cam1_idx',
            'cam0_timestamp_sec',
            'cam1_timestamp_sec',
            'time_diff_ms',
            'cam0_interval_ms',
            'cam1_interval_ms'
        ])
        
        # data rows
        for i, (cam0_idx, cam1_idx, diff) in enumerate(results['matched_pairs']):
            cam0_interval = results['cam0_intervals'][i] if i < len(results['cam0_intervals']) else np.nan
            cam1_interval = results['cam1_intervals'][cam1_idx-1] if (cam1_idx-1) < len(results['cam1_intervals']) else np.nan
                
            writer.writerow([
                i,
                cam0_idx,
                cam1_idx,
                cam0_times[cam0_idx],
                cam1_times[cam1_idx],
                diff,
                cam0_interval,
                cam1_interval
            ])

    print(f'Output CSV File saved: {output_csv}')

def print_summary(results):
    print(f"\n=== SUMMARY ===")
    print(f"Dataset Duration: {results['duration']:.2f} seconds")
    print(f"Total cam0 frames: {len(results['cam0_times'])}")
    print(f"Total cam1 frames: {len(results['cam1_times'])}")

    print(f"\n--- Frame Rates ---")
    print(f"cam0: {results['cam0_rate']:.2f} Hz (interval: {results['cam0_interval_mean']:.2f} ± {results['cam0_interval_std']:.2f} ms)")
    print(f"cam1: {results['cam1_rate']:.2f} Hz (interval: {results['cam1_interval_mean']:.2f} ± {results['cam1_interval_std']:.2f} ms)")

    print(f"\n--- Temporal Synchronization (cam1[i+1] vs cam0[i]) ---")
    print(f"Mean absolute time difference: {results['index_diff_mean']:.2f} ms")
    print(f"Std deviation: {results['index_diff_std']:.2f} ms")
    print(f"Min: {results['index_diff_min']:.2f} ms")
    print(f"Max: {results['index_diff_max']:.2f} ms")
    print(f"Median: {results['index_diff_median']:.2f} ms")

    # Interpret synchronization quality
    if results['index_diff_mean'] < 1.0:
        print(f"\n✓ EXCELLENT: Hardware synchronized (offset < 1ms)")
    elif results['index_diff_mean'] < 5.0:
        print(f"\n✓ GOOD: Well synchronized (offset < 5ms)")
    elif results['index_diff_mean'] < 20.0:
        print(f"\n⚠ FAIR: Acceptable for Kalibr --approx-sync (offset < 20ms)")
    else:
        print(f"\n✗ POOR: May cause calibration issues (offset > 20ms)")

def first_five(
        bag_path,
        cam0_topic = '/cam_sync/cam0/image_raw/debayered',
        cam1_topic = '/cam_sync/cam1/image_raw/debayered',
        n_frames = 5
):
    print("=== FIRST TIMESTAMPS FROM MCAP FILE ===\n")

    cam0_5 = []
    cam1_5 = []

    with open(bag_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=[cam0_topic, cam1_topic]
        ):
            timestamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9

            if channel.topic == cam0_topic and len(cam0_5) < n_frames:
                cam0_5.append(timestamp)
            elif channel.topic == cam1_topic and len(cam1_5) < n_frames:
                cam1_5.append(timestamp)

            if len(cam0_5) >= n_frames and len(cam1_5) >= n_frames:
                break

    print(f"cam0 first {len(cam0_5)} timestamps:")
    for i, t in enumerate(cam0_5):
        print(f"  cam0[{i}]: {t:.9f}")

    print(f"\ncam1 first {len(cam1_5)} timestamps:")
    for i, t in enumerate(cam1_5):
        print(f"  cam1[{i}]: {t:.9f}")
    
    print()


if __name__ == "__main__":
    # Configuration
    if len(sys.argv) > 1:
        bag_path = sys.argv[1]
    else:
        bag_path = '/home/sid/NeuROAM_data/merged_payload4b/alternate_skipped.mcap'

    cam0_topic = '/cam_sync/cam0/image_raw'
    cam1_topic = '/cam_sync/cam1/image_raw'

    first_five(bag_path, cam0_topic=cam0_topic, cam1_topic=cam1_topic, n_frames=5)

    output_csv = '/home/sid/CS8674/scripts/timestamp_check/alternate_mcap.csv'

    # Pipeline
    results = analyze_timestamps(
        bag_path=bag_path,
        cam0_topic='/cam_sync/cam0/image_raw',
        cam1_topic='/cam_sync/cam1/image_raw',
        output_csv=output_csv
    )