from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config = LaunchConfiguration('config')
    control = LaunchConfiguration('control')
    enabled = LaunchConfiguration('enable_controller')
    controllers = {
        'manual': 'precision_landing_controller_manual_search',
        'automatic': 'precision_landing_controller',
        'gps': 'precision_landing_controller_gps_nav',
    }
    nodes = [
        DeclareLaunchArgument('config', default_value=PathJoinSubstitution([FindPackageShare('drone_landing_service'), 'config', 'landing_real.yaml'])),
        DeclareLaunchArgument('control', default_value='manual', choices=list(controllers)),
        DeclareLaunchArgument('enable_controller', default_value='false', choices=['true', 'false']),
        Node(package='drone_landing_service', executable='target_detector', name='target_detector', parameters=[config]),
    ]
    for choice, executable in controllers.items():
        nodes.append(Node(
            package='drone_landing_service', executable=executable, name='precision_landing_controller',
            parameters=[config, {'start_on_launch': False}],
            condition=IfCondition(PythonExpression(["'", enabled, "' == 'true' and '", control, "' == '", choice, "'"])),
        ))
    nodes.append(Node(package='drone_landing_service', executable='control_arbiter', name='control_arbiter', parameters=[config], condition=IfCondition(enabled)))
    for executable in ['drone_global_position_interpreter', 'global_boat_position_interpreter']:
        nodes.append(Node(
            package='drone_landing_service', executable=executable, name=executable, parameters=[config],
            condition=IfCondition(PythonExpression(["'", enabled, "' == 'true' and '", control, "' != 'manual'"])),
        ))
    return LaunchDescription(nodes)
