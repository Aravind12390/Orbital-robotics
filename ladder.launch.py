from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = FindPackageShare('go2_cart_rig')

    xacro_file = PathJoinSubstitution(
        [pkg, 'urdf', 'go2_on_cart.urdf.xacro']
    )
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', xacro_file]), value_type=str
        )
    }

    # Gazebo Harmonic — empty world, physics running from start (-r)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
            ])
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # Publishes TF tree from robot_description
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # Spawns the combined model (ladder + trolley + robot) into Gazebo
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name',  'go2_ladder_crawl',
            '-topic', 'robot_description',
        ],
        output='screen',
    )

    return LaunchDescription([gz_sim, robot_state_publisher, spawn])
