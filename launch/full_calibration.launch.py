from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur_transform',
            executable='aruco_transform_publisher',
            name='aruco_tf'
        ),
        Node(
            package='ur_transform',
            executable='sim_camera_pose',
            name='sim_cam_tf'
        ),
    ])
