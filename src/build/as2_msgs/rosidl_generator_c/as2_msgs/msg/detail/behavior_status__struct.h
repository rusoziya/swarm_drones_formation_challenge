// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/BehaviorStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'IDLE'.
enum
{
  as2_msgs__msg__BehaviorStatus__IDLE = 0
};

/// Constant 'RUNNING'.
enum
{
  as2_msgs__msg__BehaviorStatus__RUNNING = 1
};

/// Constant 'PAUSED'.
enum
{
  as2_msgs__msg__BehaviorStatus__PAUSED = 2
};

/// Struct defined in msg/BehaviorStatus in the package as2_msgs.
typedef struct as2_msgs__msg__BehaviorStatus
{
  uint8_t status;
} as2_msgs__msg__BehaviorStatus;

// Struct for a sequence of as2_msgs__msg__BehaviorStatus.
typedef struct as2_msgs__msg__BehaviorStatus__Sequence
{
  as2_msgs__msg__BehaviorStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__BehaviorStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__STRUCT_H_
