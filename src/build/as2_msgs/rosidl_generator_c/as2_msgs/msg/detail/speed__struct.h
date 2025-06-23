// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/Speed.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__SPEED__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__SPEED__STRUCT_H_

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

/// Struct defined in msg/Speed in the package as2_msgs.
/**
  * Speed message
 */
typedef struct as2_msgs__msg__Speed
{
  /// Message header
  std_msgs__msg__Header header;
  /// speed (m/s)
  float speed;
} as2_msgs__msg__Speed;

// Struct for a sequence of as2_msgs__msg__Speed.
typedef struct as2_msgs__msg__Speed__Sequence
{
  as2_msgs__msg__Speed * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__Speed__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__SPEED__STRUCT_H_
