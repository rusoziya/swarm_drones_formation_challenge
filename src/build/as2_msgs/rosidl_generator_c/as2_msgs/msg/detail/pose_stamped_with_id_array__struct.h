// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/PoseStampedWithIDArray.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POSE_STAMPED_WITH_ID_ARRAY__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__POSE_STAMPED_WITH_ID_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'poses'
#include "as2_msgs/msg/detail/pose_stamped_with_id__struct.h"

/// Struct defined in msg/PoseStampedWithIDArray in the package as2_msgs.
/**
  * Pose Stamped with an string id array
 */
typedef struct as2_msgs__msg__PoseStampedWithIDArray
{
  as2_msgs__msg__PoseStampedWithID__Sequence poses;
} as2_msgs__msg__PoseStampedWithIDArray;

// Struct for a sequence of as2_msgs__msg__PoseStampedWithIDArray.
typedef struct as2_msgs__msg__PoseStampedWithIDArray__Sequence
{
  as2_msgs__msg__PoseStampedWithIDArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__PoseStampedWithIDArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__POSE_STAMPED_WITH_ID_ARRAY__STRUCT_H_
