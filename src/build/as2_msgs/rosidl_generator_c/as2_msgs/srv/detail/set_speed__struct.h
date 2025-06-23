// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/SetSpeed.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_SPEED__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__SET_SPEED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'speed'
#include "as2_msgs/msg/detail/speed__struct.h"

/// Struct defined in srv/SetSpeed in the package as2_msgs.
typedef struct as2_msgs__srv__SetSpeed_Request
{
  /// speed to send
  as2_msgs__msg__Speed speed;
} as2_msgs__srv__SetSpeed_Request;

// Struct for a sequence of as2_msgs__srv__SetSpeed_Request.
typedef struct as2_msgs__srv__SetSpeed_Request__Sequence
{
  as2_msgs__srv__SetSpeed_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetSpeed_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetSpeed in the package as2_msgs.
typedef struct as2_msgs__srv__SetSpeed_Response
{
  /// whether the speed has been received or not
  bool success;
} as2_msgs__srv__SetSpeed_Response;

// Struct for a sequence of as2_msgs__srv__SetSpeed_Response.
typedef struct as2_msgs__srv__SetSpeed_Response__Sequence
{
  as2_msgs__srv__SetSpeed_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetSpeed_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__SET_SPEED__STRUCT_H_
