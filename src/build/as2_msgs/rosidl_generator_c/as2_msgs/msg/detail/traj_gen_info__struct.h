// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/TrajGenInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'WAITING'.
enum
{
  as2_msgs__msg__TrajGenInfo__WAITING = 0
};

/// Constant 'EVALUATING'.
enum
{
  as2_msgs__msg__TrajGenInfo__EVALUATING = 1
};

/// Constant 'STOPPED'.
enum
{
  as2_msgs__msg__TrajGenInfo__STOPPED = 2
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'node_status'
#include "as2_msgs/msg/detail/node_status__struct.h"

/// Struct defined in msg/TrajGenInfo in the package as2_msgs.
/**
  * Message that shows the trajectory generator state
 */
typedef struct as2_msgs__msg__TrajGenInfo
{
  /// Message header
  std_msgs__msg__Header header;
  /// Node status
  as2_msgs__msg__NodeStatus node_status;
  /// Active status
  uint8_t active_status;
} as2_msgs__msg__TrajGenInfo;

// Struct for a sequence of as2_msgs__msg__TrajGenInfo.
typedef struct as2_msgs__msg__TrajGenInfo__Sequence
{
  as2_msgs__msg__TrajGenInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__TrajGenInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__STRUCT_H_
