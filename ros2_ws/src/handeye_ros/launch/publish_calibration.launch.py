from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("result_path"),
        DeclareLaunchArgument("camera_link_frame", default_value=""),
        DeclareLaunchArgument("camera_link_to_optical_path", default_value=""),
        Node(package="handeye_ros", executable="publish_calibration", output="screen", parameters=[{
            "result_path": LaunchConfiguration("result_path"),
            "camera_link_frame": LaunchConfiguration("camera_link_frame"),
            "camera_link_to_optical_path": LaunchConfiguration("camera_link_to_optical_path")}])])

