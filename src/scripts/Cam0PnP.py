import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation as R
import yaml
from cv_bridge import CvBridge
from pathlib import Path
from aprilgrid import Detector as AprilGridDetector

# mcap imports
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


@dataclass
class CalibrationConfig:
    # April grid parameters
    tag_size: float # apriltag edge length (meters)
    tag_family: str   # AprilTag family 
    tag_rows: int  # Number of tags vertically
    tag_cols: int  # Number of tags horizontally
    tag_spacing: float  # ratio of (gap between tag edges/tag size)
    min_corners: int
    ransac_reproj_threshold: float 
    ransac_confidence: float 

    # camera instrinsics
    camera_matrix: np.ndarray #3x3 K matrix
    dist_coefficients: np.ndarray # distortion coefficients 
    image_width: int
    image_height: int
    camera_name: str
    
    @classmethod
    def from_config(cls, config_data:dict, camera_name: str = 'cam0') -> 'CalibrationConfig':
        '''
        Load instrinsics from config file  
        '''
        
        calib = config_data['calibration']
        camera = config_data[camera_name]

        K = np.array(camera['camera_matrix']).reshape(3,3)
        distortion = np.array(camera['dist_coefficients']).flatten()

        return cls(
            tag_size=calib['tag_size'],
            tag_family=calib['tag_family'],
            tag_rows=calib['tag_rows'],
            tag_cols=calib['tag_cols'],
            tag_spacing=calib['tag_spacing'],
            min_corners=calib.get('min_corners', 4),
            ransac_reproj_threshold=calib.get('ransac_reproj_threshold', 8.0),
            ransac_confidence=calib.get('ransac_confidence', 0.99),
            # Camera intrinsics
            camera_matrix=K,
            dist_coefficients=distortion,
            image_width=camera['image_width'],
            image_height=camera['image_height'],
            camera_name=camera.get('camera_name', camera_name)
        )
    
    def convert_to_dict(self):
        return {
            'tag_size': float(self.tag_size),
            'tag_family': self.tag_family,
            'tag_rows': int(self.tag_rows),
            'tag_cols': int(self.tag_cols),
            'tag_spacing': float(self.tag_spacing),
            'min_corners': int(self.min_corners),
            'ransac_reproj_threshold': float(self.ransac_reproj_threshold),
            'ransac_confidence': float(self.ransac_confidence),
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coefficients': self.dist_coefficients.tolist(),
            'image_width': int(self.image_width),
            'image_height': int(self.image_height),
            'camera_name': self.camera_name
        }


@dataclass
class ImageFrame:
    '''
    container for image frame data
    '''
    timestamp: float
    image: np.ndarray
    seq: int #sequence number from ROS message
    frame_id: str #Frame ID from ROS message


@dataclass
class PoseEstimate:
    '''
    container for single pose estimate
    '''
    timestamp: float
    R: np.ndarray #3x3 rotation matrix
    t: np.ndarray #3x1 translation vector
    T: np.ndarray #4x4  homogeneous transformation matrix (T_destination_source)
    T_inv: np.ndarray 
    num_inliers: int
    reprojection_error: float
    corner_ids: List[int] #which corners were detected
    num_tags_detected: int  # Number of AprilTags detected

    ## store in dictonary format
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'R': self.R.tolist(),
            't': self.t.tolist(),
            'T': self.T.tolist(),
            'T_inv': self.T_inv.tolist(),
            'num_inliers': self.num_inliers,
            'reprojection_error': float(self.reprojection_error),
            'num_corners': len(self.corner_ids),
            'num_tags_detected': self.num_tags_detected
        }

#####################################
 
class ROSBagImageLoader:
    def __init__(self):
        self.cvbridge = CvBridge()
    
    # mcap
    def load_images_from_bag(
            self,
            bag_path: Union[str, Path],
            image_topic: str,
            max_frames: Optional[int] = None
    ) -> List[ImageFrame]:
        bag_path = Path(bag_path)

        if not bag_path.exists():
            raise FileNotFoundError(f"Bag File not found: {bag_path}")
        
        print(f"Loading images from topic '{image_topic}' in {bag_path.name}")
        
        frames = []
        seq = 0
        
        with open(bag_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            # schema = msg type definition, channel = topic info, message = raw bytes, ros_msg = deserialized ROS msg
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=[image_topic]):
                try:
                    msg_type = type(ros_msg).__name__
                    if msg_type == 'CompressedImage':
                        cv_image = self.cvbridge.compressed_imgmsg_to_cv2(ros_msg, desired_encoding='bgr8') # --> (height,width,3) shape
                    else:
                        cv_image = self.cvbridge.imgmsg_to_cv2(ros_msg, desired_encoding='bgr8') # --> (height,width,3) shape
                    
                    # timestamp from message header in seconds
                    timestamp_sec = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                    
                    frame = ImageFrame(
                        timestamp=timestamp_sec,
                        image=cv_image,
                        seq=seq,
                        frame_id=ros_msg.header.frame_id
                    )
                    frames.append(frame)
                    seq += 1
                    
                    # incase needed
                    if max_frames and len(frames) >= max_frames:
                        break
                        
                except Exception as e:
                    print(f"Failed to convert image at seq {seq}: {e}")
                    continue
        
        print(f"LOADED {len(frames)} FRAMES FROM {bag_path.name}")
        if frames:
            print(f"  Time range: {frames[0].timestamp:.3f} to {frames[-1].timestamp:.3f} seconds")

        return frames

    
#########################

# CalibrationCOnfig object 
class LoadConfig:
    # load config file from the path
    @staticmethod
    def load_config(config_path: Union[str, Path], camera_name: str='cam0') -> CalibrationConfig:
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
        return CalibrationConfig.from_config(config_data, camera_name)


#########################
    
class Camera0PnP:
    '''
    rig_frame = camera 0 frame hence
    T_rig_to_cam0 = Identity

    Notation = T_destination_source 

    For each camera 0 frame, PnP gives => T_cam0_target (target to cam0, source=target, destination=cam0)

    invert this => T_target_cam0 (source=cam0, destination=target)
    '''

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.video_writer = None

        # initialized April grid detector
        tag_family_short = self.config.tag_family.replace("tag", "t")
        self.aprilgrid_detector = AprilGridDetector(tag_family_short)
        print(f"Initialized AprilGrid detector with family: {tag_family_short}")
        print(f"Camera: {self.config.camera_name}")  # make sure it's cam 0

        # 3D points of AprilTag corners in target frame
        self.object_points = self.object_points_detected()

        # store results
        self.pose_estimates: Dict[float, PoseEstimate] = {}
        self.failed_timestamps: List[float] = []

    ############## April grid detection ###############

    # 3D positions of April tag corners
    def object_points_detected(self) -> Dict[int, np.ndarray]:
        '''
        3D coordinates of AprilTag corners in target frame (object points)
        
        Returns:
            Dictionary mapping tag_id -> 4x3 array of corner positions
        '''
        half_size = self.config.tag_size / 2  

        # 4 corners in tag local frame (x,y,z)
        local_corners = np.array([
            [ half_size, -half_size, 0],  # Bottom-right
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0],  # Top-left
            [-half_size, -half_size, 0],  # Bottom-left
        ])

        tag_points = {}  #3D points of 4 corners for each tag
        tag_id = 0  #unique key for each tag

        # tag spacing = (gap b/w tags) / tag size
        # tag center = tag size + gap b/w tags = tag size + (tag spacing * tag size) = tag size (1 + tag spacing)
        tag_center_step = self.config.tag_size * (1 + self.config.tag_spacing)   #dist between tag centers
        
        # grid of tags
        for row in range(self.config.tag_rows):
            for col in range(self.config.tag_cols):
                col_reversed = (self.config.tag_cols - 1) - col # detector gives tag0 at bottom right, hence start from last column

                center_x = col_reversed * tag_center_step
                center_y = row * tag_center_step
                center = np.array([center_x, center_y, 0])    #center of other tags w.r.t first tag at origin
                
                # local corners to world frame corners
                tag_points[tag_id] = local_corners + center   
                tag_id += 1
        
        print(f"Generated {len(tag_points)} AprilTag 3D positions for {tag_id} tags.")
        print(f"Known 3D tag IDs: {list(tag_points.keys())}")
        return tag_points
    
    # 2D positions of April tag corners 
    def detect_corners(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, List[int]]]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            '''
            detections = [detection(tag_id=0, corners=[[1], [2], [3], [4]] ), ...]
            '''
            detections = self.aprilgrid_detector.detect(gray)
            

            if len(detections) == 0:
                return None
            
            # Extract corners
            corners_2d = []
            corner_ids = []
            tag_corner_pairs = []

            h, w = gray.shape   #image height and width
            
            for detection in detections:
                tag_id = detection.tag_id

                #skip if 3D coordinates are not known
                if tag_id not in self.object_points:
                    continue

                # Extract corners and reshape if needed
                corners = np.asarray(detection.corners, dtype=np.float32)  #4x2

                if corners.ndim == 3 and corners.shape[1] == 1 and corners.shape[2] == 2:
                    corners = corners[:, 0, :]  # Remove middle dimension
                else:
                    corners = corners.reshape(-1, 2)  # Flatten to (4, 2)
                
                # validate 4 corners
                if corners.shape[0] != 4:
                    print(f"  Warning: tag {tag_id} has {corners.shape[0]} corners (expected 4), skipping")
                    continue

                ####

                # corners are within image bounds  (5 pixel safety margin for subpixel refinement) 
                valid = True 
                for corner in corners:
                    if corner[0] < 5 or corner[0] >= w - 5 or corner[1] < 5 or corner[1] >= h - 5:
                        valid = False
                        break

                if not valid:
                    continue  #skip tags with corners outside image bounds

                ####
                
                # Add each corner
                for corner_idx, corner in enumerate(corners):
                    corners_2d.append(corner) # append (x,y) position of corner 
                    
                    # calculate grid index for corner
                    grid_idx = self.get_corner_grid_idx(tag_id, corner_idx)  
                    corner_ids.append(grid_idx)  
                    
                    # tag id and it's corner indices in the tag
                    tag_corner_pairs.append((tag_id, corner_idx)) 
            
            # dont save if not enough corners detected 
            if len(corners_2d) < self.config.min_corners:
                return None
            
            # separated corners 2D positions
            corners_2d = np.array(corners_2d, dtype=np.float32)
            
            # Subpixel refinement 
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cv2.cornerSubPix(gray, corners_2d, winSize=(2, 2), zeroZone=(-1, -1), criteria=criteria)
            
            return corners_2d, corner_ids, tag_corner_pairs
            
        except Exception as e:
            print(f"  WARNING: AprilGrid detector error: {e}")
            return None
    
    # local tag corners to corners in the grid
    def get_corner_grid_idx(self, tag_id: int, corner_idx: int) -> int:
        total_cols = 2 * self.config.tag_cols  # Number of columns in cornergrid 
    
        # Tag position in the tag grid
        tag_row = tag_id // self.config.tag_cols  #which row
        tag_col = tag_id % self.config.tag_cols   #which column

        # tag_col_reversed = (self.config.tag_cols - 1) - tag_col
        
        # Base index (bottom-right corner of the tag)
        baseId = tag_row * 2 * total_cols + tag_col* 2
        
        # Corner offsets from base index
        pIdx = [
            baseId + 1,              # 0: bottom-right 
            baseId + total_cols + 1, # 1: top-right (one row up, one right)
            baseId + total_cols,     # 2: top-left (one row up)
            baseId                   # 3: bottom-left
        ]
        
        return pIdx[corner_idx]
    
    ############ PnP ###############

    def solve_pnp(
        self,
        corners_2d: np.ndarray,
        corner_ids: List[int],
        tag_corner_pairs: List[Tuple[int, int]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, float, int]]:
        '''
        Returns:
            R: 3x3 rotation matrix (target to cam0)
            t: 3x1 translation vector (target to cam0)
            num_inliers: Number of inlier correspondences
            reproj_error: Mean reprojection error in pixels
            num_tags: Number of unique tags detected
            Returns None if PnP fails
        '''

        # atleast 4 corners 
        if len(corners_2d) < self.config.min_corners:
            return None
        
        # 3D points
        object_points = []
        for tag_id,corner_idx in tag_corner_pairs:
            object_points.append(self.object_points[tag_id][corner_idx])
        
        object_points = np.array(object_points, dtype=np.float32)
        
        # PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,    #3D positions
            corners_2d,       # 2D positions
            self.config.camera_matrix,
            self.config.dist_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.config.ransac_reproj_threshold,  #default 8 pixels
            confidence=self.config.ransac_confidence   #default 0.999
        )
        
        if not success:
            return None
        
        # print(f"Raw tvec from OpenCV PnP: {tvec.flatten()}")
        # print(f"  tx (X): {tvec[0, 0]:+.6f}")
        # print(f"  ty (Y): {tvec[1, 0]:+.6f}")
        # print(f"  tz (Z): {tvec[2, 0]:+.6f}")
        
        # rotation vector to matrix
        R_mat, _ = cv2.Rodrigues(rvec) # axis angle rotation 
        t_vec = tvec.reshape(3, 1)

        # Pcam = Rmat @ Pgrid + tvec
        
        # reprojection error
        num_inliers = len(inliers) if inliers is not None else len(corners_2d)

        #inliers validation
        total_corners = len(corners_2d)
        inlier_mask = inliers.flatten() if inliers is not None else np.arange(len(corners_2d))  #corners indices that agree 

        #make sure enough corners are left after RANSAC
        if num_inliers < 16:  # At least 4 tags worth of corners
            return None

        # reproject inlier 3D points
        projected, _ = cv2.projectPoints(
            object_points[inlier_mask],
            rvec,
            tvec,
            self.config.camera_matrix,
            self.config.dist_coefficients
        )
        projected = projected.reshape(-1, 2)
        
        errors = np.linalg.norm(corners_2d[inlier_mask] - projected, axis=1)   # detected 2d and reprojected 2d
        reproj_error = np.mean(errors)

        if reproj_error > 2.0:
            return None
        
        max_error = np.max(errors)
        if max_error > 4.0:  # No single point with error > 4px
            return None
        
        # Count unique tags
        unique_tags = len(set(cid // 4 for cid in corner_ids))
        
        return R_mat, t_vec, num_inliers, reproj_error, unique_tags, inlier_mask
    
    # pipeline put together
    def process_all_frames(
        self,
        frames: List[ImageFrame],
        visualize: bool = False,
        visualize_reprojection: bool = False,
        save_worst_cases: int = 5,
        min_tags_required: int = 5,
        save_video: bool = False,
        video_path: str = "output/detection_video.mp4"
    ) -> Dict[str, any]:
        '''
        Detection + PnP
        
        '''
        print(f"Processing {len(frames)} frames for {self.config.camera_name}\n")

        frame_height, frame_width = None, None
        frames_written = 0

        # detection video
        if save_video and frames:
            h, w = frames[0].image.shape[:2]
            frame_height, frame_width = frames[0].image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            Path("output").mkdir(parents=True, exist_ok=True)
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            print(f"Video writer initialized: {video_path} ({w}x{h} @ 20fps)")
        
        num_successful = 0
        num_failed = 0
        num_insufficient_tags = 0
        worst_frames = []
        
        for i, frame in enumerate(frames):
            # detect corners
            detection_result = self.detect_corners(frame.image)
            if detection_result is None:
                self.failed_timestamps.append(frame.timestamp)
                num_failed += 1
                continue
            
            corners_2d, corner_ids, tag_corner_pairs = detection_result 

            unique_tags = len(set(tag_id for tag_id, _ in tag_corner_pairs))
            if unique_tags < min_tags_required:
                self.failed_timestamps.append(frame.timestamp)
                num_insufficient_tags += 1
                continue
            
            # Solve PnP
            pnp_result = self.solve_pnp(corners_2d, corner_ids, tag_corner_pairs)
            if pnp_result is None:
                self.failed_timestamps.append(frame.timestamp)
                num_failed += 1
                continue
            
            R_mat, t_vec, num_inliers, reproj_error, num_tags, inlier_mask = pnp_result

            if visualize_reprojection and reproj_error > 0.3:  # Only track high errors
                worst_frames.append((
                    reproj_error, 
                    i,
                    {
                        'image': frame.image,
                        'corners_2d': corners_2d,
                        'corner_ids': corner_ids,
                        'tag_corner_pairs': tag_corner_pairs,
                        'R': R_mat,
                        't': t_vec,
                        'timestamp': frame.timestamp
                    }
                ))
                worst_frames.sort(reverse=True, key=lambda x: x[0])
                worst_frames = worst_frames[:save_worst_cases]

            # Transformation matrix (T_dest_src)
            T_cam0_target = np.eye(4)

            # insert estimated R and t
            T_cam0_target[:3, :3] = R_mat  
            T_cam0_target[:3, 3:4] = t_vec 
            
            # inverse Transformation matrix to get cam trajectory
            T_target_cam0 = np.linalg.inv(T_cam0_target)
            
            # result
            pose = PoseEstimate(
                timestamp=frame.timestamp,
                R=R_mat,
                t=t_vec,
                T=T_cam0_target,
                T_inv=T_target_cam0,
                num_inliers=num_inliers,
                reprojection_error=reproj_error,
                corner_ids=corner_ids,
                num_tags_detected=num_tags
            )
            self.pose_estimates[frame.timestamp] = pose
            num_successful += 1
            
            # visualization
            if visualize or save_video:
                vis_frame = self.visualize_detection(frame.image, corners_2d, corner_ids, R_mat, t_vec,
                    save_to_video=(save_video and self.video_writer is not None)
                )
                
                if save_video and self.video_writer is not None and vis_frame is not None:
                    #validate frame dimensions
                    if vis_frame.shape[:2] == (frame_height, frame_width):
                        # bgr format for opencv
                        if len(vis_frame.shape) == 2:
                            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
                        
                        self.video_writer.write(vis_frame)
                        frames_written += 1
                    
                    else:
                        print(f"WARNING: Frame {i} dimension mismatch: "
                          f"{vis_frame.shape[:2]} vs expected ({frame_height}, {frame_width})")

            # update
            if (i + 1) % 50 == 0:
                success_rate = num_successful / (i + 1) * 100
                print(f"  Processed {i+1}/{len(frames)} frames "
                    f"(Success: {num_successful}, Failed(no PnP solution found): {num_failed}, Rate: {success_rate:.1f}%)")
        
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"  Total frames written: {frames_written}/{num_successful}")
            self.video_writer = None

        # stats
        if not self.pose_estimates:
            stats = {
                'num_successful': 0,
                'num_failed': len(self.failed_timestamps),
                'success_rate': 0.0,
                'mean_reproj_error': 0.0,
                'std_reproj_error': 0.0,
                'mean_inliers': 0.0,
                'mean_tags_detected': 0.0,
                'timestamps': []
            }
        else:
            reproj_errors = [p.reprojection_error for p in self.pose_estimates.values()]
            num_inliers = [p.num_inliers for p in self.pose_estimates.values()]
            num_tags = [p.num_tags_detected for p in self.pose_estimates.values()]
            
            total_attempts = len(self.pose_estimates) + len(self.failed_timestamps)
            
            stats = {
                'num_successful': int(len(self.pose_estimates)),
                'num_failed': int(len(self.failed_timestamps)),
                'success_rate': float(len(self.pose_estimates) / total_attempts) if total_attempts > 0 else 0.0,
                'mean_reproj_error': float(np.mean(reproj_errors)),
                'std_reproj_error': float(np.std(reproj_errors)),
                'mean_inliers': float(np.mean(num_inliers)),
                'mean_tags_detected': float(np.mean(num_tags)),
                'timestamps': sorted(self.pose_estimates.keys())
            }

        # final result
        print(f"  Total frames processed: {len(frames)}")
        print(f"  Successful: {stats['num_successful']} ({stats['success_rate']*100:.1f}%)")
        print(f"  Failed/Skipped: {stats['num_failed']}")
        print(f"  Mean reprojection error: {stats['mean_reproj_error']:.3f} pixels")
        print(f"  Mean inliers per frame: {stats['mean_inliers']:.1f}")
        print(f"  Mean tags detected per frame: {stats['mean_tags_detected']:.1f}")
            
        return stats
    
    def visualize_detection(
        self,
        image: np.ndarray,
        corners_2d: np.ndarray,
        corner_ids: List[int],
        R: np.ndarray,
        t: np.ndarray,
        save_to_video: bool = False
    ) -> np.ndarray:
        '''Draw detected AprilTag corners and coordinate axes on image'''
        vis_img = image.copy()
        
        # Draw corners grouped by tag
        tag_corners = {}
        for corner, corner_id in zip(corners_2d, corner_ids):
            tag_id = corner_id // 4
            if tag_id not in tag_corners:
                tag_corners[tag_id] = []
            tag_corners[tag_id].append(corner)
        
        # Draw each tag
        for tag_id, corners in tag_corners.items():
            corners = np.array(corners, dtype=np.int32)
            
            # Draw tag boundary
            if len(corners) == 4:
                cv2.polylines(vis_img, [corners], True, (0, 255, 0), 2)
                
                # Draw tag ID
                center = np.mean(corners, axis=0).astype(int)
                cv2.putText(vis_img, str(tag_id), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw corners
            for corner in corners:
                cv2.circle(vis_img, tuple(corner), 3, (255, 0, 0), -1)
        
        # Draw coordinate frame axes
        axis_length = 3 * self.config.tag_size  # 3 tags long
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length]   # Z-axis (blue)
        ])
        
        rvec, _ = cv2.Rodrigues(R)
        imgpts, _ = cv2.projectPoints(
            axis_points, rvec, t,
            self.config.camera_matrix,
            self.config.dist_coefficients
        )
        imgpts = imgpts.reshape(-1, 2).astype(int)
        
        origin = tuple(imgpts[0])
        cv2.line(vis_img, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X red
        cv2.line(vis_img, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y green
        cv2.line(vis_img, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z blue
        
        # window only when no video saved
        if not save_to_video:
            cv2.imshow('Camera 0 AprilTag Detection', vis_img)
            cv2.waitKey(1)

        return vis_img
    
    def get_trajectory(self) -> Dict[float, np.ndarray]:
        '''
        Cam0 trajectory as T_target_cam0(t_i) poses (cam0 -> target)
        
        Returns:
            Dictionary mapping timestamp -> 4x4 transformation matrix
        '''
        return {ts: pose.T_inv for ts, pose in self.pose_estimates.items()}
    
    
    def save_results(self, output_path: Union[str, Path]):
        '''Save pose estimates to JSON file'''
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.pose_estimates:
            stats = {
                'num_successful': 0,
                'num_failed': len(self.failed_timestamps),
                'timestamps': []
            }
        else:
            stats = {
                'num_successful': int(len(self.pose_estimates)),
                'num_failed': int(len(self.failed_timestamps)),
                'timestamps': sorted(self.pose_estimates.keys())
            }
        
        results = {
            'poses': [pose.to_dict() for pose in self.pose_estimates.values()],
            'statistics': stats,
            'config': self.config.convert_to_dict()  
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")

    def debug_tag_layout(self):
        '''Print the 3D positions of all tags to verify layout'''
        print("\n=== AprilTag 3D Layout Debug ===")
        for tag_id in sorted(self.object_points.keys()):
            corners = self.object_points[tag_id]
            center = np.mean(corners, axis=0)
            print(f"Tag {tag_id:2d}: center at ({center[0]:7.4f}, {center[1]:7.4f}, {center[2]:7.4f})")
            print(f"          corners: {corners[:, :2]}")  # Show x,y only
        print("="*50)

    def visualize_reprojection(
        self,
        image: np.ndarray,
        corners_2d: np.ndarray,
        corner_ids: List[int],
        tag_corner_pairs: List[Tuple[int, int]],
        R: np.ndarray,
        t: np.ndarray,
        inlier_mask: np.ndarray = None,
        reproj_error: float = 0.0,
        save_path: str = None,
        show_window: bool = False
    ):
        '''
        Visualize detected corners vs reprojected corners to diagnose reprojection errors
        '''
        # DEBUG: Print what tags were detected
        unique_tags = sorted(set(tag_id for tag_id, _ in tag_corner_pairs))
        print(f"\n  DEBUG: Detected tags: {unique_tags}")
        print(f"  DEBUG: Number of corners: {len(corners_2d)}")
        print(f"  DEBUG: tag_corner_pairs sample: {tag_corner_pairs[:8]}")
        
        vis_img = image.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # Get 3D object points
        object_points = []
        for tag_id, corner_idx in tag_corner_pairs:
            obj_pt = self.object_points[tag_id][corner_idx]
            object_points.append(obj_pt)
        object_points = np.array(object_points, dtype=np.float32)
        
        # DEBUG: Print sample 3D points
        print(f"  DEBUG: Sample 3D object points:")
        for i in range(min(4, len(object_points))):
            tag_id, corner_idx = tag_corner_pairs[i]
            print(f"    Tag {tag_id}, corner {corner_idx}: {object_points[i]}")
        
        # Reproject all points
        rvec, _ = cv2.Rodrigues(R)
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            t,
            self.config.camera_matrix,
            self.config.dist_coefficients
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # DEBUG: Print sample projections
        print(f"  DEBUG: Sample detected vs reprojected:")
        for i in range(min(4, len(corners_2d))):
            tag_id, corner_idx = tag_corner_pairs[i]
            detected = corners_2d[i]
            reprojected = projected_points[i]
            error = np.linalg.norm(detected - reprojected)
            print(f"    Tag {tag_id}, corner {corner_idx}:")
            print(f"      Detected: ({detected[0]:.1f}, {detected[1]:.1f})")
            print(f"      Reprojected: ({reprojected[0]:.1f}, {reprojected[1]:.1f})")
            print(f"      Error: {error:.1f} px")
        
        # Prepare inlier mask
        if inlier_mask is None:
            inlier_mask = np.ones(len(corners_2d), dtype=bool)
        elif inlier_mask.ndim == 2:  # If it's [[idx1], [idx2], ...]
            mask = np.zeros(len(corners_2d), dtype=bool)
            mask[inlier_mask.flatten()] = True
            inlier_mask = mask
        
        # Calculate per-point errors
        errors = np.linalg.norm(corners_2d - projected_points, axis=1)
        
        # Group by tags for visualization
        tag_groups = {}
        for i, (tag_id, corner_idx) in enumerate(tag_corner_pairs):
            if tag_id not in tag_groups:
                tag_groups[tag_id] = []
            tag_groups[tag_id].append(i)
        
        print(f"  DEBUG: Tags in visualization: {sorted(tag_groups.keys())}")
        
        # Draw detections and reprojections
        for i, (detected, reprojected, is_inlier, error) in enumerate(
            zip(corners_2d, projected_points, inlier_mask, errors)
        ):
            detected_pt = tuple(detected.astype(int))
            reprojected_pt = tuple(reprojected.astype(int))
            
            if is_inlier:
                # Inliers: Green detected, Blue reprojected
                cv2.circle(vis_img, detected_pt, 6, (0, 255, 0), 2)  # Green circle
                cv2.circle(vis_img, reprojected_pt, 4, (255, 0, 0), -1)  # Blue filled
                cv2.line(vis_img, detected_pt, reprojected_pt, (0, 255, 255), 1)  # Yellow line
            else:
                # Outliers: Red detected, Magenta reprojected
                cv2.circle(vis_img, detected_pt, 6, (0, 0, 255), 2)  # Red circle
                cv2.circle(vis_img, reprojected_pt, 4, (255, 0, 255), -1)  # Magenta filled
                cv2.line(vis_img, detected_pt, reprojected_pt, (128, 0, 128), 1)  # Purple line
        
        # Add tag labels
        for tag_id, indices in tag_groups.items():
            # Get center of detected corners for this tag
            tag_corners_2d = corners_2d[indices]
            center = np.mean(tag_corners_2d, axis=0).astype(int)
            cv2.putText(vis_img, f"T{tag_id}", tuple(center), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add statistics text
        num_inliers = np.sum(inlier_mask)
        num_outliers = len(inlier_mask) - num_inliers
        mean_error = np.mean(errors[inlier_mask]) if num_inliers > 0 else 0
        max_error = np.max(errors[inlier_mask]) if num_inliers > 0 else 0
        
        text_y = 30
        cv2.putText(vis_img, f"Reproj Error: {reproj_error:.2f} px", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(vis_img, f"Tags: {len(unique_tags)} | Corners: {len(corners_2d)}", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(vis_img, f"Inliers: {num_inliers} | Outliers: {num_outliers}", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(vis_img, f"Mean Error: {mean_error:.2f} px | Max: {max_error:.2f} px", 
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        legend_y = vis_img.shape[0] - 100
        cv2.putText(vis_img, "Legend:", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        cv2.circle(vis_img, (30, legend_y), 6, (0, 255, 0), 2)
        cv2.circle(vis_img, (30, legend_y), 4, (255, 0, 0), -1)
        cv2.putText(vis_img, "Inlier: Green=Detected, Blue=Reprojected", 
                    (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
        cv2.circle(vis_img, (30, legend_y), 6, (0, 0, 255), 2)
        cv2.circle(vis_img, (30, legend_y), 4, (255, 0, 255), -1)
        cv2.putText(vis_img, "Outlier: Red=Detected, Magenta=Reprojected", 
                    (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_img)
            print(f"  Reprojection visualization saved to {save_path}")
        
        if show_window:
            cv2.imshow('Reprojection Visualization', vis_img)
            cv2.waitKey(0)
        
        return vis_img, errors

## save sample frame(image) from mcap bag
def save_mcap_frame(
        bag_path: str,
        image_topic: str,
        frame_number: int=0,
        output_path: str = "mcap_frame.jpg"
):
    print(f" Saving frame {frame_number} from {bag_path} topic '{image_topic}'")

    bag_loader = ROSBagImageLoader()
    frames = bag_loader.load_images_from_bag(
        bag_path=bag_path,
        image_topic=image_topic,
        max_frames=frame_number + 1
    )

    if len(frames) <= frame_number:
        print(f" ERROR: Only {len(frames)} frames available, cannot get frame {frame_number}")
        return
    
    frame = frames[frame_number]

    cv2.imwrite(output_path, frame.image)
    print(f" Frame saved to {output_path}")
    print(f"  Timestamp: {frame.timestamp:.3f} seconds")
    print(f"  Image shape: {frame.image.shape}")
    print(f"  Frame ID: {frame.frame_id}")

    return frame.image

## aprilgrid detection visualization
def visualize_aprilgrid_detection(
    image: np.ndarray,
    corners_2d: np.ndarray,
    corner_ids: List[int],
    tag_corner_pairs: List[Tuple[int, int]],
    output_path: str = "aprilgrid_detection.jpg"
):
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()

    tag_corners = {}
    for corner, (tag_id, corner_idx) in zip(corners_2d, tag_corner_pairs):
        if tag_id not in tag_corners:
            tag_corners[tag_id] = []
        tag_corners[tag_id].append((corner, corner_idx))

    num_tags = len(tag_corners)
    num_corners = len(corners_2d)

    print(f"Tags detected:      {num_tags}")
    print(f"Tag IDs:            {sorted(tag_corners.keys())}")
    print(f"Total corners:      {num_corners}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    original_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    axes[0].imshow(original_display, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    for tag_id, corners_list in tag_corners.items():
        # Sort corners by corner_idx to get proper order
        corners_list.sort(key=lambda x: x[1])
        corners = np.array([c[0] for c in corners_list], dtype=np.int32)
        
        # Draw tag boundary (green polygon)
        if len(corners) == 4:
            cv2.polylines(vis_img, [corners], True, (0, 255, 0), 3)
        
        # Draw corners (blue circles)
        for corner in corners:
            cv2.circle(vis_img, tuple(corner), 5, (255, 0, 0), -1)
        
        # Draw tag ID at center (yellow text)
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(vis_img, f"ID:{tag_id}", tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    axes[1].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'AprilGrid Detections\n {num_tags} tags, {num_corners} corners', 
                     fontsize=14, color='green', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"Visualization saved: {output_path}")
    
    plt.close()
    
    return num_tags, num_corners

#########################

if __name__ == "__main__":
    # Configuration
    BAG_PATH = "/home/sid/NeuROAM_data/merged_payload4b/merged_decompressed_0.mcap"
    CAM0_IMAGE_TOPIC = "/cam_sync/cam0/image_raw"
    CONFIG_PATH = "/home/sid/async_vision/src/config/calibrationconfig.yaml"
    KALIBR_YAML = "/home/sid/async_vision/Kalibr/raw_data/decompressed_ros1-camchain.yaml"  #for kalibr baseline comparison

    bag_loader = ROSBagImageLoader()

    ## CAMERA 0
    config_cam0 = LoadConfig.load_config(CONFIG_PATH, camera_name='cam0')
    cam0_pnp = Camera0PnP(config_cam0)

    frames_cam0 = bag_loader.load_images_from_bag(
        bag_path=BAG_PATH,
        image_topic="/cam_sync/cam0/image_raw"
    )
    cam0_pnp.process_all_frames(
        frames = frames_cam0, 
        min_tags_required=4,
        visualize=False, 
        visualize_reprojection=False, 
        save_worst_cases=5,
        save_video=True,
        video_path="/home/sid/async_vision/src/output/cam0_detection.avi"
    )
    cam0_trajectory = cam0_pnp.get_trajectory()

    cam0_pnp.save_results("/home/sid/async_vision/src/output/cam0_trajectory_apriltag.json")

    #release memory
    del frames_cam0
    del cam0_pnp
    import gc
    gc.collect()

    ## CAMERA 1
    config_cam1 = LoadConfig.load_config(CONFIG_PATH, camera_name='cam1')
    cam1_pnp = Camera0PnP(config_cam1)  

    frames_cam1 = bag_loader.load_images_from_bag(
        bag_path=BAG_PATH,
        image_topic="/cam_sync/cam1/image_raw"
    )
    cam1_pnp.process_all_frames(
        frames=frames_cam1, 
        min_tags_required=4,
        visualize=False, 
        visualize_reprojection=False, 
        save_worst_cases=5,
        save_video=True,
        video_path="/home/sid/async_vision/src/output/cam1_detection.avi"
    )
    cam1_trajectory = cam1_pnp.get_trajectory()
    cam1_pnp.save_results("/home/sid/async_vision/src/output/cam1_trajectory_apriltag.json")

    #release memory
    del frames_cam1
    del cam1_pnp
    gc.collect()

    ## BASELINE

    # Load Kalibr baseline for comparison
    with open(KALIBR_YAML, 'r') as f:
        calib = yaml.safe_load(f)

    T_1_0_kalibr = np.array(calib['cam1']['T_cn_cnm1'])

    t_kalibr = T_1_0_kalibr[:3, 3] *1000 #Kalibr translation vector (mm)
    R_kalibr = T_1_0_kalibr[:3, :3] # Kalibr rotation matrix

    translation_errors = []
    rotation_errors = []
    baselines = []

    #baseline magnitude
    baseline_kalibr = np.linalg.norm(t_kalibr)  # mm

    common_timestamps = sorted(set(cam0_trajectory.keys()) & set(cam1_trajectory.keys()))
    print(f"Common timestamps found: {len(common_timestamps)}")

    #poses for corresponding timestamps
    for ts in common_timestamps:
        T_target_cam0 = cam0_trajectory[ts]  #cam0 pose 
        T_target_cam1 = cam1_trajectory[ts]  #cam1 pose
        
        T_1_0 = np.linalg.inv(T_target_cam1) @ T_target_cam0

        #translation error between estimated and kalibr
        t_computed = T_1_0[:3, 3] *1000  # mm
        error = np.linalg.norm(t_computed - t_kalibr) 
        translation_errors.append(error)

        #rotation error
        R_computed = T_1_0[:3, :3]

        R_error = R_kalibr.T @ R_computed #close to identity if similar
        angle_error = np.abs(R.from_matrix(R_error).magnitude())  # radians
        angle_error_deg = np.degrees(angle_error)
        rotation_errors.append(angle_error_deg)

        #baseline magnitude check
        baseline_computed = np.linalg.norm(t_computed)
        baselines.append(baseline_computed)

print(f"\nBaseline (translation norm):")
print(f"  Kalibr reference: {baseline_kalibr:.2f} mm")
print(f"  Computed - Mean: {np.mean(baselines):.2f} mm, Std: {np.std(baselines):.2f} mm")
print(f"Rotation error - Mean: {np.mean(rotation_errors):.2f}°, Std: {np.std(rotation_errors):.2f}°")
print(f"Translation error - Mean: {np.mean(translation_errors):.2f} mm, Std: {np.std(translation_errors):.2f} mm")
        
        
