// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/FollowTargetInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__STRUCT_H_

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
  as2_msgs__msg__FollowTargetInfo__WAITING = 0
};

/// Constant 'RUNNING'.
enum
{
  as2_msgs__msg__FollowTargetInfo__RUNNING = 1
};

/// Constant 'END'.
enum
{
  as2_msgs__msg__FollowTargetInfo__END = 2
};

/// Constant 'UNSET'.
/**
  * Mode when the follow mode is not set
 */
enum
{
  as2_msgs__msg__FollowTargetInfo__UNSET = 0
};

/// Constant 'PICKUP'.
enum
{
  as2_msgs__msg__FollowTargetInfo__PICKUP = 1
};

/// Constant 'UNPICK'.
enum
{
  as2_msgs__msg__FollowTargetInfo__UNPICK = 2
};

/// Constant 'DYNAMIC_LAND'.
enum
{
  as2_msgs__msg__FollowTargetInfo__DYNAMIC_LAND = 3
};

/// Constant 'DYNAMIC_FOLLOWER'.
enum
{
  as2_msgs__msg__FollowTargetInfo__DYNAMIC_FOLLOWER = 4
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/FollowTargetInfo in the package as2_msgs.
/**
  * Message that encodes the possible follow target info supported in Aerostack2.
 */
typedef struct as2_msgs__msg__FollowTargetInfo
{
  /// Message header
  std_msgs__msg__Header header;
  /// Follow status
  int8_t follow_status;
  /// Follow mode
  int8_t follow_mode;
} as2_msgs__msg__FollowTargetInfo;

// Struct for a sequence of as2_msgs__msg__FollowTargetInfo.
typedef struct as2_msgs__msg__FollowTargetInfo__Sequence
{
  as2_msgs__msg__FollowTargetInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__FollowTargetInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__STRUCT_H_
