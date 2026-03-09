"""
FrameSkipper: Simulates asynchronous stereo camera data from synchronized MCAP recordings.

Reads a synchronized stereo MCAP bag and produces a modified bag where the two cameras
appear unsynchronized. Supports three skip patterns: ALTERNATING (interleaved frames),
DIFFERENT_RATES (cam0 full rate, cam1 half rate), and OFFSET (fixed temporal offset on cam1).
Output can be written as MCAP or ROS2 .db3 (sqlite3) format.
"""

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from pathlib import Path
import yaml
from typing import Dict, List, Optional
from enum import Enum

class Skippattern(Enum):
    ALTERNATING = "alternating"   # cam0: 0,2,4,6   cam1: 1,3,5,7
    DIFFERENT_RATES = "diff_rates"  # cam0: 20Hz, cam1:10Hz
    OFFSET = "offset" # X ms offset to cam1 timestamps

class FrameSkipper:
    def __init__(
            self,
            input_bag: str,
            output_bag: str,
            camera_topics: Dict[str, str],
            skip_pattern: Skippattern = Skippattern.OFFSET,
            output_format: str = "mcap",  # "mcap" or "sqlite3"
            storage_id: str='',
            offset_ms: float = -40.0
    ):
        self.input_bag = input_bag
        self.output_bag = output_bag
        self.camera_topics = camera_topics
        self.skip_pattern = skip_pattern
        self.output_format = output_format
        self.storage_id = storage_id
        self.offset_ns = int(offset_ms * 1e6)  # convert to nanosec

        # frame contours for each camera topic
        self.frame_counters = {topic: 0 for topic in camera_topics.values()}

        #stats
        self.stats = {
            'total_frames': {topic: 0 for topic in camera_topics.values()},
            'kept_frames': {topic:0 for topic in camera_topics.values()},
            'dropped_frames': {topic:0 for topic in camera_topics.values()},
            'offset_applied': 0
        }

        # store topic metadata for writer
        self.topic_metadata = {}
        self.schema_map = {}

    def which_frame_to_keep(self, topic:str, frame_idx: int) -> bool:
        '''
        which frame to keep based on pattern
        '''
        #which camera this topic is from
        camera_id = None

        for i, (cam_name, cam_topic) in enumerate(self.camera_topics.items()):
            if cam_topic == topic:
                camera_id = i
                break

        if camera_id is None:
            # not a camera topic, don't change anything
            return True
        
        if self.skip_pattern == Skippattern.ALTERNATING:
            # cam0 (camera_id=0): keep even frames [0,2,4,6...]
            # cam1 (camera_id=1): keep odd frames [1,3,5,7...]
            if camera_id == 0:
                return frame_idx % 2 == 0  # keep even indices for cam0
            else:
                return frame_idx % 2 == 1  # keep odd indices for cam1
        
        if self.skip_pattern == Skippattern.DIFFERENT_RATES:
            # cam0: keep all frames (20Hz -> 20Hz)
            # cam1: keep every other frame (20Hz -> 10Hz)
            if camera_id == 0:
                return True  #keep all cam0 frames
            else:
                return frame_idx % 2 == 0  # keep even frames for cam1
            
        if self.skip_pattern == Skippattern.OFFSET:
            # keep all frames, just modify timestamps
            return True
            
        return True
    
    # cam1 frames 50 ms apart but timestamps are only 10 ms apart
    def apply_offset(self, msg, topic):
        # offset only to cam1 topic
        cam1_topic = list(self.camera_topics.values())[1]

        if topic == cam1_topic and self.skip_pattern in [Skippattern.OFFSET]:
            # get message type from schema
            if topic in self.topic_metadata:
                msg_type = self.topic_metadata[topic]
                MsgType = get_message(msg_type)

                #deserialize
                message = deserialize_message(msg, MsgType)

                # check if message has header field
                if not hasattr(message, 'header'):
                    print(f"Warning: {topic} message type {msg_type} has no header field, skipping offset")
                    return msg

                #modify header timestamp
                new_sec = message.header.stamp.sec  #as it is
                new_nanosec = message.header.stamp.nanosec + self.offset_ns  

                #add 1 sec and adjust nanosec if nanosec overflows by adding offset
                if new_nanosec >= 1e9:
                    new_sec += int(new_nanosec // 1e9)
                    new_nanosec = int(new_nanosec % 1e9)

                # borrow sec and subtract nanosec if subtracting made nanosec negative
                elif new_nanosec < 0:
                    new_sec -= 1
                    new_nanosec = int(1e9 + new_nanosec)

                #update timestamp
                message.header.stamp.sec = new_sec
                message.header.stamp.nanosec = int(new_nanosec)
                
                #re-serialize
                modified_data = serialize_message(message)
                self.stats['offset_applied'] += 1

                return modified_data
            
        # original data if no offset
        return msg

    # ------------------------------------------------------------------ #
    #  Writer helpers — MCAP vs rosbag2 (sqlite3)                        #
    # ------------------------------------------------------------------ #

    def _create_rosbag2_writer(self, channel_map):
        """Set up a rosbag2_py SequentialWriter for .db3 output."""
        storage_options = rosbag2_py.StorageOptions(
            uri=self.output_bag,
            storage_id="sqlite3",
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)

        # register every topic once
        registered_topics = set()
        for channel_id, channel in channel_map.items():
            if channel.topic not in registered_topics:
                msg_type = self.topic_metadata.get(channel.topic, "")
                topic_info = rosbag2_py.TopicMetadata(
                    name=channel.topic,
                    type=msg_type,
                    serialization_format="cdr",
                )
                writer.create_topic(topic_info)
                registered_topics.add(channel.topic)

        return writer

    def _write_rosbag2_message(self, writer, topic, timestamp_ns, data):
        """Write a single message through rosbag2_py."""
        writer.write(topic, data, timestamp_ns)

    # ------------------------------------------------------------------ #
    #  Main processing loop                                              #
    # ------------------------------------------------------------------ #

    def run_skipper(self):
        print(f"Loading bag: {self.input_bag}")
        print(f"Skip pattern: {self.skip_pattern}")
        print(f"Output format: {self.output_format}")
        print(f"Camera topics: {self.camera_topics}")

        #reader
        with open(self.input_bag, "rb") as input_file:
            reader = make_reader(input_file, decoder_factories=[DecoderFactory()])
            
            # get summary for schemas and channels
            summary = reader.get_summary()
            
            # collect all schemas
            for schema_id, schema in summary.schemas.items():
                self.schema_map[schema_id] = schema
            
            # collect all channels
            channel_map = {}
            for channel_id, channel in summary.channels.items():
                channel_map[channel_id] = channel
                # store topic metadata
                if channel.schema_id in self.schema_map:
                    schema = self.schema_map[channel.schema_id]
                    self.topic_metadata[channel.topic] = schema.name

            print(f"Auto-detected output storage: {self.output_format}")

            # --- set up the appropriate writer ---
            if self.output_format == "sqlite3":
                writer = self._create_rosbag2_writer(channel_map)
                use_rosbag2 = True
            else:
                from mcap.writer import Writer as McapWriter

                #writer
                output_file = open(self.output_bag, "wb")
                writer = McapWriter(output_file)
                writer.start(profile="ros2", library="python mcap-ros2")
                use_rosbag2 = False

                # register all schemas
                schema_id_map = {}
                for schema_id, schema in self.schema_map.items():
                    new_schema_id = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data
                    )
                    schema_id_map[schema_id] = new_schema_id

                # register all channels
                channel_id_map = {}
                for channel_id, channel in channel_map.items():
                    new_channel_id = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_id_map[channel.schema_id],
                        metadata=channel.metadata
                    )
                    channel_id_map[channel_id] = new_channel_id

            #process messages
            total_messages = 0
            kept_messages = 0

            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                data = message.data
                
                # ## MODIFY TIMESTAMPS
                # # for cam1 topic, timestamp - 40ms (for ALTERNATE), comment out for other SKIP PATTERNS
                # if topic == list(self.camera_topics.values())[1]:
                #     timestamp = message.log_time - int(40 * 1e6)

                # else:
                #     # cam0 or other topics
                #     timestamp = message.log_time

                timestamp = message.log_time
                total_messages += 1

                #validate if it's camera topic
                if topic in self.camera_topics.values():
                    frame_idx = self.frame_counters[topic]
                    
                    ###########
                    if frame_idx == 0:
                        msg_type = self.topic_metadata.get(topic)
                        if msg_type:
                            MsgType = get_message(msg_type)
                            msg_obj = deserialize_message(data, MsgType)
                            print(f"\n{topic} first frame:")
                            print(f"  Message type: {msg_type}")
                            print(f"  Encoding: {getattr(msg_obj, 'encoding', 'N/A')}")
                            print(f"  Size: {getattr(msg_obj, 'width', 'N/A')}x{getattr(msg_obj, 'height', 'N/A')}")
                            print(f"  Data length: {len(getattr(msg_obj, 'data', []))} bytes")
                    
                    ###########
                    
                    self.frame_counters[topic] += 1
                    self.stats['total_frames'][topic] += 1

                    # decide whether to keep this frame based on skip_pattern
                    if self.which_frame_to_keep(topic, frame_idx):
                        modified_data = self.apply_offset(data, topic)   # apply offset if needed (for OFFSET pattern)
                        
                        if use_rosbag2:
                            self._write_rosbag2_message(writer, topic, timestamp, modified_data)
                        else:
                            writer.add_message(
                                channel_id=channel_id_map[channel.id],
                                log_time=timestamp,
                                data=modified_data,
                                publish_time=message.publish_time
                            )
                        self.stats['kept_frames'][topic] += 1
                        kept_messages += 1
                    else:
                        self.stats['dropped_frames'][topic] += 1

                else:
                    # not camera topic, keep the data
                    if use_rosbag2:
                        self._write_rosbag2_message(writer, topic, timestamp, data)
                    else:
                        writer.add_message(
                            channel_id=channel_id_map[channel.id],
                            log_time=timestamp,
                            data=data,
                            publish_time=message.publish_time
                        )
                    kept_messages += 1
                    continue

                # progress
                if total_messages % 1000 == 0:
                    print(f"Processed {total_messages} messages...", end='\r')

            # --- close writer ---
            if use_rosbag2:
                del writer  # rosbag2_py cleans up on deletion
            else:
                writer.finish()
                output_file.close()

        print(f"\n Processed {total_messages} messages total")
        print(f"Kept {kept_messages} messages in output bag")

        #stats
        self.print_stats()

    def print_stats(self):
        print("\n FRAME SKIP STATISTICS ")

        for cam_name, topic in self.camera_topics.items():
            total = self.stats['total_frames'][topic]
            kept = self.stats['kept_frames'][topic]
            dropped = self.stats['dropped_frames'][topic]
            
            print(f"\n{cam_name} ({topic}):")
            print(f"  Total frames:   {total}")
            print(f"  Kept frames:    {kept}")
            print(f"  Dropped frames: {dropped}")
            if total > 0:
                print(f"  Drop rate:      {(dropped/total)*100:.1f}%")

        if self.skip_pattern == Skippattern.OFFSET:
            print(f"\nTimestamp offset applied to {self.stats['offset_applied']} cam1 frames")

    def save_metadata(self, metadata_path: str):
        """Save processing metadata for later reference."""
        metadata = {
            'input_bag': self.input_bag,
            'output_bag': self.output_bag,
            'skip_pattern': self.skip_pattern.value,
            'output_format': self.output_format,
            'camera_topics': self.camera_topics,
            'statistics': self.stats
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"\nMetadata saved to: {metadata_path}")

###########

if __name__ == "__main__":

    ## Skip Pattern - ALTERNATING
    ## Output: ROS2 .db3 bag 
    skipper = FrameSkipper(
        input_bag = '/home/sid/NeuROAM_data/merged_payload4b/merged_decompressed_0.mcap',
        output_bag = '/home/sid/NeuROAM_data/merged_payload4b/alternate_skipped',  # rosbag2_py creates this as a directory
        camera_topics={
            'cam0': '/cam_sync/cam0/image_raw',
            'cam1': '/cam_sync/cam1/image_raw'
        },
        skip_pattern=Skippattern.ALTERNATING,
        output_format='sqlite3',
    )

    skipper.run_skipper()
    skipper.save_metadata('/home/sid/NeuROAM_data/merged_payload4b/alternate_skipped.yaml')