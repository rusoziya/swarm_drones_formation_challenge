// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/NodeStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__NODE_STATUS__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__NODE_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'UNCONFIGURED'.
enum
{
  as2_msgs__msg__NodeStatus__UNCONFIGURED = 0
};

/// Constant 'INACTIVE'.
enum
{
  as2_msgs__msg__NodeStatus__INACTIVE = 1
};

/// Constant 'ACTIVE'.
enum
{
  as2_msgs__msg__NodeStatus__ACTIVE = 2
};

/// Constant 'FINALIZED'.
enum
{
  as2_msgs__msg__NodeStatus__FINALIZED = 3
};

/// Struct defined in msg/NodeStatus in the package as2_msgs.
/**
  * Message that shows the node status
 */
typedef struct as2_msgs__msg__NodeStatus
{
  /// node status
  int8_t status;
} as2_msgs__msg__NodeStatus;

// Struct for a sequence of as2_msgs__msg__NodeStatus.
typedef struct as2_msgs__msg__NodeStatus__Sequence
{
  as2_msgs__msg__NodeStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__NodeStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__NODE_STATUS__STRUCT_H_
