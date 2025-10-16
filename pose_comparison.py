import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
    """
    Convert a pose (position + rotation vector) to a 4x4 transformation matrix.
    
    Args:
        pose: A tuple or list of 6 elements (x, y, z, rx, ry, rz),
              where (rx, ry, rz) is the rotation vector.
    
    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """
    # Extract position and rotation vector from the pose
    pos = np.array(pose[:3])
    rot_vec = np.array(pose[3:])
    
    # Create a rotation object from the rotation vector
    # The `from_rotvec` method handles the conversion to a rotation matrix
    rotation = R.from_rotvec(rot_vec)
    rot_matrix = rotation.as_matrix()
    
    # Create the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = pos
    
    return T

def compute_transformation(pose1, pose2):
    """
    Compute the transformation from pose1 to pose2.
    
    Returns:
        T_1_to_2: 4x4 transformation matrix such that pose2 = T_1_to_2 @ pose1
    """
    T1 = pose_to_matrix(pose1)
    T2 = pose_to_matrix(pose2)
    
    # Transformation from pose1 to pose2: T_1_to_2 = T2 @ inv(T1)
    T_1_to_2 = T2 @ np.linalg.inv(T1)
    
    return T_1_to_2

if __name__ == "__main__":
    # Define two poses (position + quaternion)
    pose1 = [-0.9095423, -0.39405365,  0.21069481, -1.90193083, -2.21189091,  0.82120518]
    pose2 = [-0.910, -0.40014, 0.20200, 2.031, 2.376, -0.905]

    # Compute transformation matrices
    T1 = pose_to_matrix(pose1)
    T2 = pose_to_matrix(pose2)
    T_1_to_2 = compute_transformation(pose1, pose2)
    
    print("Pose 1 Transformation Matrix:")
    print(T1)
    print("\nPose 2 Transformation Matrix:")
    print(T2)
    print("\nTransformation from Pose 1 to Pose 2:")
    print(T_1_to_2)
    
    # Verify: T2 should equal T_1_to_2 @ T1
    print("\nVerification (should be close to Pose 2):")
    print(T_1_to_2 @ T1)

"""
example1:
[[ 0.99979666 -0.0153076   0.01312668 -0.008662  ]
 [ 0.01518681  0.99984186  0.00925289  0.00349354]
 [-0.01326625 -0.00905166  0.99987103 -0.02871346]
 [ 0.          0.          0.          1.        ]]

example2:
[[ 0.99991318 -0.00582551  0.01181892 -0.00532241]
 [ 0.00571998  0.99994365  0.00894348 -0.00279034]
 [-0.01187035 -0.0088751   0.99989016 -0.02296552]
 [ 0.          0.          0.          1.        ]]
 """