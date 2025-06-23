// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/ControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROL_MODE__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__CONTROL_MODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'NONE'.
enum
{
  as2_msgs__msg__ControlMode__NONE = 0
};

/// Constant 'YAW_ANGLE'.
/**
  * Yaw angle control mode (rad)
 */
enum
{
  as2_msgs__msg__ControlMode__YAW_ANGLE = 1
};

/// Constant 'YAW_SPEED'.
/**
  * Yaw speed control mode (rad/s)
 */
enum
{
  as2_msgs__msg__ControlMode__YAW_SPEED = 2
};

/// Constant 'UNSET'.
/**
  * mode when the controller is not set
 */
enum
{
  as2_msgs__msg__ControlMode__UNSET = 0
};

/// Constant 'HOVER'.
/**
  * mode when the controller is in hover mode
 */
enum
{
  as2_msgs__msg__ControlMode__HOVER = 1
};

/// Constant 'POSITION'.
/**
  * x,   y , z  refs
 */
enum
{
  as2_msgs__msg__ControlMode__POSITION = 2
};

/// Constant 'SPEED'.
/**
  * vx, vy , vz refs
 */
enum
{
  as2_msgs__msg__ControlMode__SPEED = 3
};

/// Constant 'SPEED_IN_A_PLANE'.
/**
  * vx, vy , z refs
 */
enum
{
  as2_msgs__msg__ControlMode__SPEED_IN_A_PLANE = 4
};

/// Constant 'ATTITUDE'.
/**
  * roll, pitch, yaw  refs
 */
enum
{
  as2_msgs__msg__ControlMode__ATTITUDE = 5
};

/// Constant 'ACRO'.
/**
  * roll, pitch, yawrate  refs
 */
enum
{
  as2_msgs__msg__ControlMode__ACRO = 6
};

/// Constant 'TRAJECTORY'.
/**
  * x, y z, vx, vy, vz, ax, ay, az refs
 */
enum
{
  as2_msgs__msg__ControlMode__TRAJECTORY = 7
};

/// Constant 'UNDEFINED_FRAME'.
enum
{
  as2_msgs__msg__ControlMode__UNDEFINED_FRAME = 0
};

/// Constant 'LOCAL_ENU_FRAME'.
/**
  * local coordinates (use this by default)
 */
enum
{
  as2_msgs__msg__ControlMode__LOCAL_ENU_FRAME = 1
};

/// Constant 'BODY_FLU_FRAME'.
/**
  * body coordinates
 */
enum
{
  as2_msgs__msg__ControlMode__BODY_FLU_FRAME = 2
};

/// Constant 'GLOBAL_LAT_LONG_ASML'.
/**
  * gnss wcoordinates
 */
enum
{
  as2_msgs__msg__ControlMode__GLOBAL_LAT_LONG_ASML = 3
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/ControlMode in the package as2_msgs.
/**
  * Message that encodes the possible control modes supported in Aerostack2.
 */
typedef struct as2_msgs__msg__ControlMode
{
  /// Message header
  std_msgs__msg__Header header;
  /// Yaw mode
  int8_t yaw_mode;
  /// Control mode
  int8_t control_mode;
  /// Reference frame
  int8_t reference_frame;
} as2_msgs__msg__ControlMode;

// Struct for a sequence of as2_msgs__msg__ControlMode.
typedef struct as2_msgs__msg__ControlMode__Sequence
{
  as2_msgs__msg__ControlMode * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__ControlMode__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__CONTROL_MODE__STRUCT_H_
