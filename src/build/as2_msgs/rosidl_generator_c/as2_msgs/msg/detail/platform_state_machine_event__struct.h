// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/PlatformStateMachineEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'EMERGENCY'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__EMERGENCY = -1
};

/// Constant 'ARM'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__ARM = 0
};

/// Constant 'DISARM'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__DISARM = 1
};

/// Constant 'TAKE_OFF'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__TAKE_OFF = 2
};

/// Constant 'TOOK_OFF'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__TOOK_OFF = 3
};

/// Constant 'LAND'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__LAND = 4
};

/// Constant 'LANDED'.
enum
{
  as2_msgs__msg__PlatformStateMachineEvent__LANDED = 5
};

/// Struct defined in msg/PlatformStateMachineEvent in the package as2_msgs.
/**
  * Platform events that actives aerial platform state machine 
 */
typedef struct as2_msgs__msg__PlatformStateMachineEvent
{
  /// Platform state machine event
  int8_t event;
} as2_msgs__msg__PlatformStateMachineEvent;

// Struct for a sequence of as2_msgs__msg__PlatformStateMachineEvent.
typedef struct as2_msgs__msg__PlatformStateMachineEvent__Sequence
{
  as2_msgs__msg__PlatformStateMachineEvent * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__PlatformStateMachineEvent__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__STRUCT_H_
