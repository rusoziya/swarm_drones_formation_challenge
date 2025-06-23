import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ziyar/project_gazebo_ws/src/challenge_multi_drone/install/scenario_obstacle_viz'
