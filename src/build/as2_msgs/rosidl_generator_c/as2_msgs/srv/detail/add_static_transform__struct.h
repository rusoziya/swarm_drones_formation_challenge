// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/AddStaticTransform.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'frame_id'
// Member 'child_frame_id'
#include "rosidl_runtime_c/string.h"
// Member 'transform'
#include "geometry_msgs/msg/detail/transform__struct.h"

/// Struct defined in srv/AddStaticTransform in the package as2_msgs.
typedef struct as2_msgs__srv__AddStaticTransform_Request
{
  /// Parent frame frame id
  rosidl_runtime_c__String frame_id;
  /// Child frame id
  rosidl_runtime_c__String child_frame_id;
  /// Transform
  geometry_msgs__msg__Transform transform;
} as2_msgs__srv__AddStaticTransform_Request;

// Struct for a sequence of as2_msgs__srv__AddStaticTransform_Request.
typedef struct as2_msgs__srv__AddStaticTransform_Request__Sequence
{
  as2_msgs__srv__AddStaticTransform_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__AddStaticTransform_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/AddStaticTransform in the package as2_msgs.
typedef struct as2_msgs__srv__AddStaticTransform_Response
{
  /// whether the transform has been set or not
  bool success;
} as2_msgs__srv__AddStaticTransform_Response;

// Struct for a sequence of as2_msgs__srv__AddStaticTransform_Response.
typedef struct as2_msgs__srv__AddStaticTransform_Response__Sequence
{
  as2_msgs__srv__AddStaticTransform_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__AddStaticTransform_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__STRUCT_H_
