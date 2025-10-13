import os
import numpy as np

from third_party.ur_robot.pickobj import solve_pickbox_with_grasp_real_robot as solver
# Example usage for real robot pick and place

# Robot configuration
ROBOT_IP = "192.168.50.144"

# Object configuration (example pose in world coordinates)
object_pose = np.array([
    [-9.165771007537841797e-01, -3.827639818191528320e-01, -1.156642660498619080e-01, -1.471330225467681885e-01],
    [1.414123922586441040e-01, -5.808652639389038086e-01, 8.016222119331359863e-01, 8.609708398580551147e-02],
    [-3.740174174308776855e-01, 7.183921337127685547e-01, 5.865353941917419434e-01, 6.474282145500183105e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
])

# Grasp configuration
grasp_file = None

print("Starting real robot pick and place demo...")
print(f"Robot IP: {ROBOT_IP}")
print(f"Object pose: {object_pose}")
print(f"Grasp file: {grasp_file}")

success = solver(
    robot_ip=ROBOT_IP,
    object_pose=object_pose,
    grasp_file_path=grasp_file,
    grasp_idx=3,
    debug=True,
    approach_distance=0.1,
    lift_height=0.1,
    return_to_home=True
)

print("Demo finished.")