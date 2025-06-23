#!/bin/bash

usage() {
    echo "  options:"
    echo "      -s: simulated, choices: [true | false]"
    echo "      -v: visualisation (rviz2), choices: [true | false]"
    echo "      -m: multi agent, choices: [true | false]"
    echo "      -e: estimator_type, choices: [ground_truth, raw_odometry, mocap_pose]"
    echo "      -r: record rosbag"
    echo "      -t: launch keyboard teleoperation"
    echo "      -n: drone namespace, default is cf0"
    echo "      -w: world file"
    echo "      -a: auto run mission"
}

# Arg parser
while getopts "sve:mrtnaw:" opt; do
  case ${opt} in
    s )
      simulated="true"
      ;;
    v )
      visualise="true"
      ;;
    m )
      swarm="true"
      ;;
    e )
      estimator_plugin="${OPTARG}"
      ;;
    r )
      record_rosbag="true"
      ;;
    t )
      launch_keyboard_teleop="true"
      ;;
    n )
      drone_namespace="${OPTARG}"
      ;;
    w )
      world_file="${OPTARG}"
      ;;
    a )
      auto_run="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[swrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

source utils/tools.bash

# Shift optional args
shift $((OPTIND -1))

## DEFAULTS
simulated=${simulated:="false"}  # default ign_gz
if [[ ${simulated} == "false" && -z ${estimator_plugin} ]]; then
  echo "Error: when -s is false, -e argument must be set" 1>&2
  usage
  exit 1
fi

swarm=${swarm:="false"}
visualise=${visualise:="false"}
estimator_plugin=${estimator_plugin:="ground_truth"}  # default ign_gz
record_rosbag=${record_rosbag:="false"}
launch_keyboard_teleop=${launch_keyboard_teleop:="false"}
drone_namespace=${drone_namespace:="cf"}
auto_run=${auto_run:="false"}
world_file=${world_file:="sim_config/world.json"}

if [[ ${swarm} == "true" ]]; then
  num_drones=2
  simulation_config="sim_config/world_swarm.json"
else
  num_drones=1
  simulation_config="${world_file}"
fi

export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:"$(pwd)/models"
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:"$(pwd)/models"

# Generate the list of drone namespaces
drone_ns=()
for ((i=0; i<${num_drones}; i++)); do
  drone_ns+=("$drone_namespace$i")
done

for ns in "${drone_ns[@]}"
do
  if [[ ${ns} == ${drone_ns[0]} ]]; then
    base_launch="true"
  else
    base_launch="false"
  fi 

  tmuxinator start -n ${ns} -p utils/session.yml drone_namespace=${ns} base_launch=${base_launch}  estimator_plugin=${estimator_plugin} simulation=${simulated} simulation_config=${simulation_config} auto_run=${auto_run}&
  wait
done

if [[ ${estimator_plugin} == "mocap_pose" ]]; then
  tmuxinator start -n mocap -p utils/mocap.yml &
  wait
fi

if [[ ${record_rosbag} == "true" ]]; then
  tmuxinator start -n rosbag -p utils/rosbag.yml drone_namespace=$(list_to_string "${drone_ns[@]}") &
  wait
fi

if [[ ${launch_keyboard_teleop} == "true" ]]; then
  tmuxinator start -n keyboard_teleop -p utils/keyboard_teleop.yml simulation=true drone_namespace=$(list_to_string "${drone_ns[@]}") &
  wait
fi

if [[ ${simulated} == "true" ]]; then
  tmuxinator start -n gazebo -p utils/gazebo.yml simulation_config=${simulation_config} &
  wait
fi

if [[ ${visualise} == "true" ]]; then
  tmuxinator start -n viz -p utils/visualisation.yml &
  wait
fi

# Attach to tmux session ${drone_ns[@]}, window 0
tmux attach-session -t ${drone_ns[0]}:mission
