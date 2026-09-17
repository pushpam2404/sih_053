from setuptools import setup

package_name = "drdo_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Moving-object detection for the DRDO ID26053 foveated 2.5D map",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dynamic_obstacle_node = drdo_perception.dynamic_obstacle_node:main",
        ],
    },
)
