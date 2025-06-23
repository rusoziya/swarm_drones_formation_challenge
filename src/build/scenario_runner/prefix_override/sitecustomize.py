import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ziyar/project_gazebo_ws/src/install/scenario_runner'
