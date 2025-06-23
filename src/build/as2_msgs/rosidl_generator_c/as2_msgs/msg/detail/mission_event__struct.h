// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/MissionEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__MISSION_EVENT__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__MISSION_EVENT__STRUCT_H_

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
// Member 'data'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/MissionEvent in the package as2_msgs.
/**
  * Message for trigger mission events
 */
typedef struct as2_msgs__msg__MissionEvent
{
  /// Message header
  std_msgs__msg__Header header;
  /// (Optional) data to send with the trigger
  rosidl_runtime_c__String data;
} as2_msgs__msg__MissionEvent;

// Struct for a sequence of as2_msgs__msg__MissionEvent.
typedef struct as2_msgs__msg__MissionEvent__Sequence
{
  as2_msgs__msg__MissionEvent * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__MissionEvent__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__MISSION_EVENT__STRUCT_H_
