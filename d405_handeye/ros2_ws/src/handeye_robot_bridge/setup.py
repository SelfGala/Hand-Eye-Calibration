from setuptools import setup
from glob import glob
setup(name="handeye_robot_bridge", version="0.1.0", packages=["handeye_robot_bridge"],
      data_files=[("share/ament_index/resource_index/packages", ["resource/handeye_robot_bridge"]),
                  ("share/handeye_robot_bridge", ["package.xml"]),
                  ("share/handeye_robot_bridge/launch", glob("launch/*.launch.py"))],
      install_requires=["setuptools"], zip_safe=True,
      maintainer="Handeye Maintainer", maintainer_email="maintainer@example.invalid", license="Apache-2.0",
      description="On-demand end-pose service",
      entry_points={"console_scripts": ["pose_service = handeye_robot_bridge.node:main",
                                       "mock_pose_service = handeye_robot_bridge.mock:main"]})
