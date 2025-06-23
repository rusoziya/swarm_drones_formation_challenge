// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/Thrust.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__THRUST__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__THRUST__STRUCT_H_

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

/// Struct defined in msg/Thrust in the package as2_msgs.
/**
  * Message for encoding the desired thrust value
 */
typedef struct as2_msgs__msg__Thrust
{
  /// Message header
  std_msgs__msg__Header header;
  /// Thrust (N)
  float thrust;
  /// Thrust normalized [0,1]
  float thrust_normalized;
} as2_msgs__msg__Thrust;

// Struct for a sequence of as2_msgs__msg__Thrust.
typedef struct as2_msgs__msg__Thrust__Sequence
{
  as2_msgs__msg__Thrust * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__Thrust__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__THRUST__STRUCT_H_
