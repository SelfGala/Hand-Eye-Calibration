from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("config_path"), Node(package="handeye_robot_bridge",
        executable="pose_service", output="screen", parameters=[{"config_path": LaunchConfiguration("config_path")}])])

