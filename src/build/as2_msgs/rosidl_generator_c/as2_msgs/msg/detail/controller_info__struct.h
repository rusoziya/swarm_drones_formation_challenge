// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/ControllerInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__STRUCT_H_

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
// Member 'input_control_mode'
// Member 'output_control_mode'
#include "as2_msgs/msg/detail/control_mode__struct.h"

/// Struct defined in msg/ControllerInfo in the package as2_msgs.
/**
  * Message that shows the controller state and the current input_control_mode
 */
typedef struct as2_msgs__msg__ControllerInfo
{
  /// Message header
  std_msgs__msg__Header header;
  /// Input control mode
  as2_msgs__msg__ControlMode input_control_mode;
  /// Output control mode
  as2_msgs__msg__ControlMode output_control_mode;
} as2_msgs__msg__ControllerInfo;

// Struct for a sequence of as2_msgs__msg__ControllerInfo.
typedef struct as2_msgs__msg__ControllerInfo__Sequence
{
  as2_msgs__msg__ControllerInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__ControllerInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__STRUCT_H_
