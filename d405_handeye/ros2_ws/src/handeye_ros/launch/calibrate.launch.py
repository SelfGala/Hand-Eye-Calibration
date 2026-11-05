from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("config_path"),
        Node(package="handeye_ros", executable="handeye_calibrator", namespace="handeye", output="screen",
             arguments=["--config", LaunchConfiguration("config_path")])])

