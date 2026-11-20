from setuptools import setup
from glob import glob
setup(name="handeye_ros", version="0.1.0", packages=["handeye_ros"],
      data_files=[("share/ament_index/resource_index/packages", ["resource/handeye_ros"]),
                  ("share/handeye_ros", ["package.xml"]),
                  ("share/handeye_ros/launch", glob("launch/*.launch.py"))],
      install_requires=["setuptools"], zip_safe=True,
      maintainer="Handeye Maintainer", maintainer_email="maintainer@example.invalid", license="Apache-2.0",
      description="Eye-in-hand native desktop application",
      entry_points={"console_scripts": ["handeye_calibrator = handeye_ros.app:main",
                                       "publish_calibration = handeye_ros.publish:main"]})
