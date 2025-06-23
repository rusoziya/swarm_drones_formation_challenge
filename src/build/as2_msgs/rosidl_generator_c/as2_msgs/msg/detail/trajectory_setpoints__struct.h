// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/TrajectorySetpoints.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJECTORY_SETPOINTS__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__TRAJECTORY_SETPOINTS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'setpoints'
#include "as2_msgs/msg/detail/trajectory_point__struct.h"

/// Struct defined in msg/TrajectorySetpoints in the package as2_msgs.
/**
  * Definition of a point of a trajectory
 */
typedef struct as2_msgs__msg__TrajectorySetpoints
{
  /// Message header with the frame_id of the point
  std_msgs__msg__Header header;
  /// Array of setpoints of the vehicle in the frame_id frame
  as2_msgs__msg__TrajectoryPoint__Sequence setpoints;
} as2_msgs__msg__TrajectorySetpoints;

// Struct for a sequence of as2_msgs__msg__TrajectorySetpoints.
typedef struct as2_msgs__msg__TrajectorySetpoints__Sequence
{
  as2_msgs__msg__TrajectorySetpoints * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__TrajectorySetpoints__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__TRAJECTORY_SETPOINTS__STRUCT_H_
