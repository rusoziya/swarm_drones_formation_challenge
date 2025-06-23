// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/GimbalControl.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'POSITION_MODE'.
enum
{
  as2_msgs__msg__GimbalControl__POSITION_MODE = 0
};

/// Constant 'SPEED_MODE'.
enum
{
  as2_msgs__msg__GimbalControl__SPEED_MODE = 1
};

// Include directives for member types
// Member 'target'
#include "geometry_msgs/msg/detail/vector3_stamped__struct.h"

/// Struct defined in msg/GimbalControl in the package as2_msgs.
/**
  * Gimbal Control message definition
 */
typedef struct as2_msgs__msg__GimbalControl
{
  uint8_t control_mode;
  /// x: roll y: pitch z: yaw
  geometry_msgs__msg__Vector3Stamped target;
} as2_msgs__msg__GimbalControl;

// Struct for a sequence of as2_msgs__msg__GimbalControl.
typedef struct as2_msgs__msg__GimbalControl__Sequence
{
  as2_msgs__msg__GimbalControl * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__GimbalControl__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__STRUCT_H_
