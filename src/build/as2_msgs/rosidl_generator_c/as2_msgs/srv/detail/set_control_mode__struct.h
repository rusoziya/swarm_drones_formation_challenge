// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/SetControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'control_mode'
#include "as2_msgs/msg/detail/control_mode__struct.h"

/// Struct defined in srv/SetControlMode in the package as2_msgs.
typedef struct as2_msgs__srv__SetControlMode_Request
{
  /// Control mode to set
  as2_msgs__msg__ControlMode control_mode;
} as2_msgs__srv__SetControlMode_Request;

// Struct for a sequence of as2_msgs__srv__SetControlMode_Request.
typedef struct as2_msgs__srv__SetControlMode_Request__Sequence
{
  as2_msgs__srv__SetControlMode_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetControlMode_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetControlMode in the package as2_msgs.
typedef struct as2_msgs__srv__SetControlMode_Response
{
  /// whether the control mode has been set or not
  bool success;
} as2_msgs__srv__SetControlMode_Response;

// Struct for a sequence of as2_msgs__srv__SetControlMode_Response.
typedef struct as2_msgs__srv__SetControlMode_Response__Sequence
{
  as2_msgs__srv__SetControlMode_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetControlMode_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__STRUCT_H_
