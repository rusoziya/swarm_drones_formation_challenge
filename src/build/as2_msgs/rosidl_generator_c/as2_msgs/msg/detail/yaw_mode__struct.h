// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/YawMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__YAW_MODE__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__YAW_MODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'KEEP_YAW'.
/**
  * Keep the current yaw angle
 */
enum
{
  as2_msgs__msg__YawMode__KEEP_YAW = 0
};

/// Constant 'PATH_FACING'.
/**
  * Yaw angle is aligned with the path
 */
enum
{
  as2_msgs__msg__YawMode__PATH_FACING = 1
};

/// Constant 'FIXED_YAW'.
/**
  * Yaw angle is fixed to a given angle
 */
enum
{
  as2_msgs__msg__YawMode__FIXED_YAW = 2
};

/// Constant 'YAW_FROM_TOPIC'.
/**
  * Yaw angle is set by a topic
 */
enum
{
  as2_msgs__msg__YawMode__YAW_FROM_TOPIC = 3
};

/// Constant 'YAW_FROM_ORIENTATION'.
/**
  * Yaw angle is set by pose orientation
 */
enum
{
  as2_msgs__msg__YawMode__YAW_FROM_ORIENTATION = 4
};

/// Constant 'YAW_TO_FRAME'.
/**
  * Yaw angle is set to face the used frame
 */
enum
{
  as2_msgs__msg__YawMode__YAW_TO_FRAME = 5
};

/// Struct defined in msg/YawMode in the package as2_msgs.
/**
  * Yaw goal
 */
typedef struct as2_msgs__msg__YawMode
{
  /// Yaw mode
  uint8_t mode;
  /// Fixed yaw (rad)
  float angle;
} as2_msgs__msg__YawMode;

// Struct for a sequence of as2_msgs__msg__YawMode.
typedef struct as2_msgs__msg__YawMode__Sequence
{
  as2_msgs__msg__YawMode * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__YawMode__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__YAW_MODE__STRUCT_H_
