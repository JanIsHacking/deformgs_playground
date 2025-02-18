import time
import numpy as np
from xarm.wrapper import XArmAPI

def move_xarm_to_keypoints(ip: str, keypoints: np.ndarray, speed: float = 100, wait: bool = True):
    """
    Moves a xArm 7 robotic arm through a sequence of Cartesian keypoints.
    
    :param ip: IP address of the xArm.
    :param keypoints: (N, 4) NumPy array where each row is (x, y, z, gripper_state).
    :param speed: Movement speed (default: 100 mm/s).
    :param wait: Whether to wait for each move to finish before proceeding to the next.
    """
    if keypoints.shape[1] != 4:
        raise ValueError("Keypoints should have shape (N, 4), representing (x, y, z, gripper_state).")

    arm = XArmAPI(ip)
    arm.connect()
    
    arm.motion_enable(True)
    arm.set_mode(0)  # Position control mode
    arm.set_state(0)  # Set arm to running state

    for i, (x, y, z, gripper_state) in enumerate(keypoints):
        print(f"Moving to point {i+1}: ({x}, {y}, {z}) | Gripper: {'Closed' if gripper_state else 'Open'}")
        arm.set_position(x, y, z, speed=speed, wait=wait)
        arm.set_gripper_position(800 if gripper_state else 0, wait=True)  # 800 = closed, 0 = open
        time.sleep(0.5)

    print("Finished executing all keypoints.")
    arm.disconnect()


def move_xarm_to_joint_states(ip: str, joint_states: np.ndarray, speed: float = 30, wait: bool = True):
    """
    Moves a xArm 7 robotic arm through a sequence of joint states.

    :param ip: IP address of the xArm.
    :param joint_states: (N, 8) NumPy array where each row is (j1, j2, j3, j4, j5, j6, j7, gripper_state).
    :param speed: Movement speed in degrees per second (default: 30).
    :param wait: Whether to wait for each move to finish before proceeding to the next.
    """
    if joint_states.shape[1] != 8:
        raise ValueError("Joint states should have shape (N, 8), representing (j1, j2, j3, j4, j5, j6, j7, gripper_state).")

    arm = XArmAPI(ip)
    arm.connect()
    
    arm.motion_enable(True)
    arm.set_mode(1)  # Joint control mode
    arm.set_state(0)  # Set arm to running state

    for i, (*joints, gripper_state) in enumerate(joint_states):
        print(f"Moving to joint state {i+1}: {joints} | Gripper: {'Closed' if gripper_state else 'Open'}")
        arm.set_servo_angle(joints, speed=speed, wait=wait)
        arm.set_gripper_position(800 if gripper_state else 0, wait=True)
        time.sleep(0.5)

    print("Finished executing all joint states.")
    arm.disconnect()

def execute_keypoints(arm1, arm2, keypoints1, keypoints2, wait_time=0.5):
    """
    Executes keypoints sequentially for a bimanual xArm 7 setup in parallel.
    :param arm1: First XArm instance
    :param arm2: Second XArm instance
    :param keypoints1: (N, 8) array for the first arm (7 joint angles + gripper)
    :param keypoints2: (N, 8) array for the second arm (7 joint angles + gripper)
    :param wait_time: Delay between keypoints
    """
    assert keypoints1.shape == keypoints2.shape, "Keypoint arrays must have the same shape"
    assert keypoints1.shape[1] == 8, "Each keypoint must have 8 values (7 joints + gripper)"
    
    N = keypoints1.shape[0]
    for i in range(N):
        q1, gripper1 = keypoints1[i, :7], keypoints1[i, 7]
        q2, gripper2 = keypoints2[i, :7], keypoints2[i, 7]
        
        # Move arms to the joint positions in parallel
        arm1.set_servo_angle(angle=q1, wait=False)
        arm2.set_servo_angle(angle=q2, wait=False)
        
        # Wait until both arms finish moving
        while arm1.get_is_moving() or arm2.get_is_moving():
            time.sleep(0.01)
        
        # Control grippers in parallel
        arm1.set_gripper_position(gripper1, wait=False)
        arm2.set_gripper_position(gripper2, wait=False)
        
        # Wait until both grippers finish moving
        while arm1.get_is_moving() or arm2.get_is_moving():
            time.sleep(0.01)
        
        time.sleep(wait_time)


if __name__ == 'main':
    keypoints = np.load("motion_data.npz")
    move_xarm_to_keypoints("192.168.1.100", keypoints)
    move_xarm_to_joint_states("192.168.1.100", keypoints)

    