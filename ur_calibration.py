import numpy as np

from third_party.ur_robot.calibration.pose_capture import CalibrationDataCollector as CDC
from third_party.ur_robot.calibration.caculator import caculate
from third_party.AprilTag.scripts.apriltag_image import apriltag_image

# Configuration
robot_ip = "192.168.50.144"
output_dir = None  # Will auto-generate with timestamp

# Create collector
collector = CDC(
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
    move_time=2.5,   # Time to move between poses
    settle_time=0.5  # Time to settle before capturing
)

tag_poses = apriltag_image(data['image_files'], output_images=False, display_images=False)
tool_poses = data['tcp_poses']

tag_H_tool, camera_H_base = caculate(tool_poses, tag_poses)

print("Calibration Results:")
print("Tag to Tool Transformation (tag_H_tool):")
print(tag_H_tool)
print("\nCamera to Robot Base Transformation (camera_H_base):")
print(camera_H_base)
