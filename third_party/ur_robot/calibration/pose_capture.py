"""
Eye-to-hand calibration data collection script.

This script moves the robot to multiple poses and captures images with the RealSense camera
at each position. The TCP poses and corresponding images are saved for calibration processing.
"""

import os
import time
import json
import numpy as np
import pyrealsense2 as rs
import cv2
from datetime import datetime
from typing import List, Dict
from ..controller import RoboticArm


class CalibrationDataCollector:
    """
    Collects robot TCP poses and corresponding camera images for eye-to-hand calibration.
    """
    
    def __init__(self, 
                 robot_ip: str,
                 output_dir: str = None,
                 image_width: int = 640,
                 image_height: int = 480,
                 fps: int = 30):
        """
        Initialize the calibration data collector.
        
        Args:
            robot_ip: IP address of the UR robot
            output_dir: Directory to save calibration data (default: ./calibration_data_TIMESTAMP)
            image_width: Width of captured images
            image_height: Height of captured images
            fps: Frame rate for RealSense camera
        """
        self.robot_ip = robot_ip
        
        # Create output directory with timestamp
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./calibration_data/{timestamp}"
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        # RealSense camera settings
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        
        # Storage for collected data
        self.tcp_poses = []
        self.image_files = []
        self.timestamps = []
        
    def setup_camera(self):
        """
        Initialize and configure the RealSense camera.
        
        Returns:
            pipeline: RealSense pipeline
        """
        print("Initializing RealSense camera...")
        
        # Create pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure only color stream
        config.enable_stream(rs.stream.color, 
                           self.image_width, 
                           self.image_height, 
                           rs.format.bgr8, 
                           self.fps)
        
        # Start pipeline
        pipeline.start(config)
        
        # Warm up camera
        print("Warming up camera...")
        for _ in range(30):
            pipeline.wait_for_frames()
        
        print("Camera ready!")
        return pipeline
    
    def capture_image(self, pipeline, index: int) -> str:
        """
        Capture RGB image from RealSense camera.
        
        Args:
            pipeline: RealSense pipeline
            index: Image index for filename
            
        Returns:
            str: RGB image filename
        """
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            raise RuntimeError("Failed to capture frame")
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Save image
        rgb_filename = f"image_{index:04d}.png"
        rgb_path = os.path.join(self.output_dir, "images", rgb_filename)
        
        cv2.imwrite(rgb_path, color_image)
        
        print(f"Saved: {rgb_filename}")
        
        return os.path.abspath(rgb_path)
    
    def get_predefined_joint_poses(self) -> List[np.ndarray]:
        """
        Generate a set of predefined joint poses for calibration.
        These poses should provide good coverage of the workspace and various orientations.
        
        Returns:
            List of joint position arrays (in radians)
        """
        # Define poses in degrees, then convert to radians
        poses_deg = [
            [10, -60, 100, -150, -75, -10],
            [15, -60, 100, -160, -83, 16],
            [10, -60, 80, -110, -80, -20],
            [10, -70, 96, -125, -78, -4],
            [3, -58, 78, -110, -80, -50],
            [1, -48, 75, -124, -81, -70],
            [3, -42, 70, -124, -60, -70],
            [8, -40, 82, -170, -61, -23],
            [15, -46, 88, -170, -67, 15]
        ]
        
        # Convert to radians
        poses_rad = [np.array(pose) / 180.0 * np.pi for pose in poses_deg]
        
        return poses_rad
    
    def collect_calibration_data(self, 
                                 num_poses: int = None,
                                 move_time: float = 3.0,
                                 settle_time: float = 1.0,
                                 custom_poses: List[np.ndarray] = None) -> Dict:
        """
        Main function to collect calibration data by moving robot and capturing images.
        
        Args:
            num_poses: Number of poses to collect (default: use all predefined poses)
            move_time: Time to move between poses (seconds)
            settle_time: Time to wait after reaching pose before capturing (seconds)
            custom_poses: Optional list of custom joint poses to use instead of predefined
            
        Returns:
            Dictionary containing collected data
        """
        print("\n" + "="*60)
        print("Starting Eye-to-Hand Calibration Data Collection")
        print("="*60 + "\n")
        
        # Get poses to visit
        if custom_poses is not None:
            joint_poses = custom_poses
        else:
            joint_poses = self.get_predefined_joint_poses()
        
        if num_poses is not None:
            joint_poses = joint_poses[:num_poses]
        
        print(f"Will collect data at {len(joint_poses)} different poses")
        
        # Setup camera
        pipeline = self.setup_camera()
        
        try:
            # Initialize robot
            print("\nInitializing robot...")
            with RoboticArm(
                robot_ip=self.robot_ip,
                frequency=125,
                max_pos_speed=0.1,  # Safe speeds
                max_rot_speed=0.3,
                tcp_offset=0.13,
                init_joints=True,
                use_gripper=False,  # Don't need gripper for calibration
                gripper_port=63352
            ) as robot:
                
                print("Robot ready!\n")
                
                # Collect data at each pose
                for i, joint_pose in enumerate(joint_poses):
                    print(f"\n--- Pose {i+1}/{len(joint_poses)} ---")
                    print(f"Target joints (deg): {joint_pose * 180 / np.pi}")
                    
                    # Move to target pose
                    print("Moving to target pose...")
                    target_time = time.time() + move_time
                    robot.exec_joint_action(joint_pose, target_time)
                    
                    # Wait for movement to complete
                    time.sleep(move_time)
                    
                    # Wait for robot to settle
                    print(f"Settling for {settle_time}s...")
                    time.sleep(settle_time)
                    
                    # Get current TCP pose
                    tcp_pose = robot.get_state()['ActualTCPPose']
                    
                    if tcp_pose is None:
                        print("Warning: Could not get TCP pose, skipping this position")
                        continue
                    
                    print(f"TCP pose: {tcp_pose}")
                    
                    # Capture image
                    print("Capturing image...")
                    image_file = self.capture_image(pipeline, i)
                    
                    # Store data
                    self.tcp_poses.append(tcp_pose)
                    self.image_files.append(image_file)
                    self.timestamps.append(time.time())
                    
                    print(f"✓ Data point {i+1} collected successfully")

                target_time = time.time() + move_time
                robot.return_to_home(target_time)
                time.sleep(move_time)

                print("\n" + "="*60)
                print(f"Data collection complete! Collected {len(self.tcp_poses)} poses")
                print("="*60)
                
        finally:
            # Stop camera
            pipeline.stop()
            print("\nCamera stopped")
        
        # Save collected data
        self.save_data()
        
        return self.get_collected_data()
    
    def save_data(self):
        """
        Save collected TCP poses and metadata to JSON file.
        """
        data = {
            'num_poses': len(self.tcp_poses),
            'tcp_poses': [pose.tolist() for pose in self.tcp_poses],
            'image_files': self.image_files,
            'timestamps': self.timestamps,
            'camera_settings': {
                'width': self.image_width,
                'height': self.image_height,
                'fps': self.fps
            }
        }
        
        json_path = os.path.join(self.output_dir, 'calibration_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nCalibration data saved to: {json_path}")
        
        # Also save TCP poses as numpy array for convenience
        np_path = os.path.join(self.output_dir, 'tcp_poses.npy')
        np.save(np_path, np.array(self.tcp_poses))
        print(f"TCP poses saved to: {np_path}")
    
    def get_collected_data(self) -> Dict:
        """
        Get the collected calibration data.
        
        Returns:
            Dictionary containing TCP poses, image files, and timestamps
        """
        return {
            'tcp_poses': self.tcp_poses,
            'image_files': self.image_files,
            'timestamps': self.timestamps,
            'output_dir': self.output_dir
        }


def main():
    """
    Main function to run calibration data collection.
    """
    # Configuration
    robot_ip = "192.168.50.144"
    output_dir = None  # Will auto-generate with timestamp
    
    # Create collector
    collector = CalibrationDataCollector(
        robot_ip=robot_ip,
        output_dir=output_dir,
        image_width=640,
        image_height=480,
        fps=30
    )
    
    # Collect data at predefined poses
    # You can specify num_poses to collect fewer samples (useful for testing)
    data = collector.collect_calibration_data(
        num_poses=None,  # None = use all predefined poses (~15 poses)
        move_time=3.0,   # Time to move between poses
        settle_time=1.0  # Time to settle before capturing
    )
    
    print("\n" + "="*60)
    print("CALIBRATION DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Output directory: {data['output_dir']}")
    print(f"Number of poses collected: {len(data['tcp_poses'])}")
    print(f"TCP poses shape: {np.array(data['tcp_poses']).shape}")
    print("\nNext steps:")
    print("1. Process the images to detect the calibration pattern")
    print("2. Extract camera poses from the detected patterns")
    print("3. Use the TCP poses and camera poses for hand-eye calibration")
    print("="*60)


if __name__ == "__main__":
    main()