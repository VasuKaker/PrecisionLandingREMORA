import os
from pathlib import Path
from setuptools import find_packages, setup

package_name = "drone_landing_service"
package_root = Path(__file__).resolve().parent


def collect(pattern: str) -> list[str]:
    return [str(p.relative_to(package_root)) for p in package_root.glob(pattern) if p.is_file()]


def collect_tree(root: str) -> list[tuple[str, list[str]]]:
    entries = []
    for directory, _, filenames in os.walk(package_root / root):
        files = [
            str((Path(directory) / name).relative_to(package_root))
            for name in filenames
        ]
        if files:
            rel_dir = Path(directory).relative_to(package_root)
            entries.append((os.path.join("share", package_name, str(rel_dir)), files))
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ament index marker
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # launch + config
        (os.path.join("share", package_name, "launch"), collect("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), collect("config/*.yaml")),
    ] + collect_tree("sim") + collect_tree("assets"),
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Prince Patel",
    maintainer_email="princep@mit.edu",
    description="ROS 2 precision landing service for MAVROS + ArUco detection.",
    license="MIT",
    # Each entry maps an executable name -> module:main
    entry_points={
        "console_scripts": [
            "drone_global_position_interpreter = drone_landing_service.drone_global_position_interpreter:main",
            "global_boat_position_interpreter = drone_landing_service.global_boat_position_interpreter:main",
            "camera_image_interpreter = drone_landing_service.camera_image_interpreter:main",
            "target_detector = drone_landing_service.target_detector:main",
            "square_detector_experimental = drone_landing_service.square_detector_experimental:main",
            "control_arbiter = drone_landing_service.control_arbiter:main",
            "precision_landing_controller = drone_landing_service.precision_landing_controller:main",
            "precision_landing_controller_manual_search = drone_landing_service.precision_landing_controller_manual_search:main",
            "precision_landing_controller_gps_nav = drone_landing_service.precision_landing_controller_gps_nav:main",
        ],
    },
)
