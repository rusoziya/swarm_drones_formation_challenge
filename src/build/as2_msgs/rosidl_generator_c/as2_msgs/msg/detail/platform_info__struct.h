// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/PlatformInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__STRUCT_H_

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
// Member 'status'
#include "as2_msgs/msg/detail/platform_status__struct.h"
// Member 'current_control_mode'
#include "as2_msgs/msg/detail/control_mode__struct.h"

/// Struct defined in msg/PlatformInfo in the package as2_msgs.
/**
  * Message that shows the platform status and the current control mode
 */
typedef struct as2_msgs__msg__PlatformInfo
{
  /// Message header
  std_msgs__msg__Header header;
  /// Whether the platform is connected or not
  bool connected;
  /// Whether the platform is armed or not
  bool armed;
  /// Whether the offboard mode is set or not
  bool offboard;
  /// Platform status
  as2_msgs__msg__PlatformStatus status;
  /// Platform control mode
  as2_msgs__msg__ControlMode current_control_mode;
} as2_msgs__msg__PlatformInfo;

// Struct for a sequence of as2_msgs__msg__PlatformInfo.
typedef struct as2_msgs__msg__PlatformInfo__Sequence
{
  as2_msgs__msg__PlatformInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__PlatformInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__STRUCT_H_
