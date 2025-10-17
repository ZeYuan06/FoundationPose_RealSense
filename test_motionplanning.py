import os
import numpy as np

from third_party.ur_robot.pickobj import solve_pickbox_with_grasp_real_robot as solver
# Example usage for real robot pick and place

# Robot configuration
ROBOT_IP = "192.168.50.144"

# Object configuration (example pose in world coordinates)
# object_pose = np.array([
#     [3.978361961947520231e-01, -8.962551733857466729e-01, 1.960939615964889526e-01, -1.372670848710098521e-02],
#     [-3.781486452050906166e-01, -3.549238799920181320e-01, -8.550044298171997070e-01, 1.590928885612021837e-01],
#     [8.359006981879935072e-01, 2.659990305269441802e-01, -4.801193177700042725e-01, 6.913155316120819238e-01],
#     [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
# ])

object_pose = np.array([
    [7.959139035617336655e-02, -9.958564171631361095e-01, -4.399234179182270887e-02, -1.745283760091142777e-01],
    [-9.957724262082755518e-01, -7.739957498635373356e-02, -4.946375195332133184e-02, 6.626171253949182660e-02],
    [4.585375818784344359e-02, 4.774323829423212512e-02, -9.978067926106102270e-01, 7.167009088028472030e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
])

# Grasp configuration
grasp_file = "/home/zy3722/work/FoundationPose/third_party/ur_robot/grasps/clamp_grasps.npz"

print("Starting real robot pick and place demo...")
print(f"Robot IP: {ROBOT_IP}")
print(f"Object pose: {object_pose}")
print(f"Grasp file: {grasp_file}")

success = solver(
    robot_ip=ROBOT_IP,
    object_pose=object_pose,
    grasp_file_path=grasp_file,
    grasp_idx=0,
    debug=True,
    approach_distance=0.1,
    lift_height=0.1,
    return_to_home=True
)

print("Demo finished.")