// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/Acro.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ACRO__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__ACRO__STRUCT_H_

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
// Member 'angular_rates'
// Member 'thrust'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/Acro in the package as2_msgs.
/**
  * Message for RPY rates and thrust (ACRO)
 */
typedef struct as2_msgs__msg__Acro
{
  /// Message header
  std_msgs__msg__Header header;
  /// Roll-, pitch-, yaw-rate around body axes
  geometry_msgs__msg__Vector3 angular_rates;
  /// Thrust expressed in the body frame.
  /// For a fixed-wing, usually the x-component is used.
  /// For a multi-rotor, usually the z-component is used.
  /// Set all un-used components to 0.
  geometry_msgs__msg__Vector3 thrust;
} as2_msgs__msg__Acro;

// Struct for a sequence of as2_msgs__msg__Acro.
typedef struct as2_msgs__msg__Acro__Sequence
{
  as2_msgs__msg__Acro * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__Acro__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__ACRO__STRUCT_H_
