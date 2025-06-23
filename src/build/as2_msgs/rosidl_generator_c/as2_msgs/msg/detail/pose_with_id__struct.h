// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/PoseWithID.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'id'
#include "rosidl_runtime_c/string.h"
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in msg/PoseWithID in the package as2_msgs.
/**
  * A Pose with an string id
 */
typedef struct as2_msgs__msg__PoseWithID
{
  /// Identification string
  rosidl_runtime_c__String id;
  /// Pose
  geometry_msgs__msg__Pose pose;
} as2_msgs__msg__PoseWithID;

// Struct for a sequence of as2_msgs__msg__PoseWithID.
typedef struct as2_msgs__msg__PoseWithID__Sequence
{
  as2_msgs__msg__PoseWithID * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__PoseWithID__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__STRUCT_H_
