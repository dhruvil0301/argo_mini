from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('argo_mini')
    urdf_file = os.path.join(pkg_share, 'urdf', 'argo_mini.urdf')
    
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()
    
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
        ),
    ])
