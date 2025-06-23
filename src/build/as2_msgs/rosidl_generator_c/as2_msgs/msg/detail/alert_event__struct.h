// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/AlertEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ALERT_EVENT__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__ALERT_EVENT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'KILL_SWITCH'.
/**
  * if value is < 0 then this alert will be handled by the platform directly
 */
enum
{
  as2_msgs__msg__AlertEvent__KILL_SWITCH = -1
};

/// Constant 'EMERGENCY_HOVER'.
enum
{
  as2_msgs__msg__AlertEvent__EMERGENCY_HOVER = -2
};

/// Constant 'EMERGENCY_LAND'.
enum
{
  as2_msgs__msg__AlertEvent__EMERGENCY_LAND = -3
};

/// Constant 'INFO_ALERT'.
/**
  * 0 value is used for ping or info alerts
 */
enum
{
  as2_msgs__msg__AlertEvent__INFO_ALERT = 0
};

/// Constant 'FORCE_HOVER'.
/**
  * if value is > 0 then this alert will be handled by the AS2 framework
 */
enum
{
  as2_msgs__msg__AlertEvent__FORCE_HOVER = 1
};

/// Constant 'FORCE_LAND'.
enum
{
  as2_msgs__msg__AlertEvent__FORCE_LAND = 2
};

// Include directives for member types
// Member 'description'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/AlertEvent in the package as2_msgs.
/**
  * Message that encodes different alert Events that can be handled by AS2 framework
 */
typedef struct as2_msgs__msg__AlertEvent
{
  int8_t alert;
  /// Further description of the alert, for debugging purposes mainly
  rosidl_runtime_c__String description;
} as2_msgs__msg__AlertEvent;

// Struct for a sequence of as2_msgs__msg__AlertEvent.
typedef struct as2_msgs__msg__AlertEvent__Sequence
{
  as2_msgs__msg__AlertEvent * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__AlertEvent__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__ALERT_EVENT__STRUCT_H_
