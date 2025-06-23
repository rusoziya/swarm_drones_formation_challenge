// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/TrajectoryPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'position'
// Member 'twist'
// Member 'acceleration'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/TrajectoryPoint in the package as2_msgs.
/**
  * Definition of a point of a trajectory
 */
typedef struct as2_msgs__msg__TrajectoryPoint
{
  /// Position of the vehicle in the frame_id frame
  geometry_msgs__msg__Vector3 position;
  /// Twist of the vehicle in the frame_id frame
  geometry_msgs__msg__Vector3 twist;
  /// Acceleration of the vehicle in the frame_id frame
  geometry_msgs__msg__Vector3 acceleration;
  /// Yaw angle of the vehicle (rad) in the frame_id frame
  float yaw_angle;
} as2_msgs__msg__TrajectoryPoint;

// Struct for a sequence of as2_msgs__msg__TrajectoryPoint.
typedef struct as2_msgs__msg__TrajectoryPoint__Sequence
{
  as2_msgs__msg__TrajectoryPoint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__TrajectoryPoint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__STRUCT_H_
