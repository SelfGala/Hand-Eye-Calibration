from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context, *args, **kwargs):
    arguments = ["--config", LaunchConfiguration("config_path").perform(context)]
    resume = LaunchConfiguration("resume").perform(context).strip()
    if resume:
        arguments += ["--resume", resume]
    return [Node(package="handeye_ros", executable="handeye_calibrator", namespace="handeye", output="screen",
                 arguments=arguments)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config_path"),
        DeclareLaunchArgument("resume", default_value="", description="已有会话目录；留空则新建。填 latest 则继续最近一次可恢复会话"),
        OpaqueFunction(function=_nodes)])

