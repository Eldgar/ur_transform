from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur_transform',
            executable='real_camera_pose',
            name='real_camera_pose_node',
            output='screen'
        ),
        Node(
            package='ur_transform',
            executable='real_aruco_transform_publisher',
            name='aruco_transform_publisher',
            output='screen'
        ),
        Node(
            package='ur_transform',
            executable='collision_objects',
            name='collision_objects_loader',
            output='screen'
        ),
        # Node(
        #     package='ur_transform',
        #     executable='real_aruco_pose',
        #     name='real_aruco_pose_node',
        #     output='screen'
        # ),
        # static transform from rg2_gripper_base_link → center_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='rg2_gripper_to_center_link_broadcaster',
            output='screen',
            arguments=[
                '0.225', '0.0', '0.0',
                '0', '0', '0', '1',
                'rg2_gripper_base_link',
                'center_link',
            ],
        ),
    ])


