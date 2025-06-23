// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/PlatformStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_STATUS__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_STATUS__STRUCT_H_

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
  as2_msgs__msg__PlatformStatus__EMERGENCY = -1
};

/// Constant 'DISARMED'.
enum
{
  as2_msgs__msg__PlatformStatus__DISARMED = 0
};

/// Constant 'LANDED'.
enum
{
  as2_msgs__msg__PlatformStatus__LANDED = 1
};

/// Constant 'TAKING_OFF'.
enum
{
  as2_msgs__msg__PlatformStatus__TAKING_OFF = 2
};

/// Constant 'FLYING'.
enum
{
  as2_msgs__msg__PlatformStatus__FLYING = 3
};

/// Constant 'LANDING'.
enum
{
  as2_msgs__msg__PlatformStatus__LANDING = 4
};

/// Struct defined in msg/PlatformStatus in the package as2_msgs.
/**
  * Platform states that an aerial platform can have
 */
typedef struct as2_msgs__msg__PlatformStatus
{
  /// platform status
  int8_t state;
} as2_msgs__msg__PlatformStatus;

// Struct for a sequence of as2_msgs__msg__PlatformStatus.
typedef struct as2_msgs__msg__PlatformStatus__Sequence
{
  as2_msgs__msg__PlatformStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__PlatformStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_STATUS__STRUCT_H_
