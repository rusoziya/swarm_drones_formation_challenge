// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/SetOrigin.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_ORIGIN__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__SET_ORIGIN__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'origin'
#include "geographic_msgs/msg/detail/geo_point__struct.h"

/// Struct defined in srv/SetOrigin in the package as2_msgs.
typedef struct as2_msgs__srv__SetOrigin_Request
{
  /// origin to set
  geographic_msgs__msg__GeoPoint origin;
} as2_msgs__srv__SetOrigin_Request;

// Struct for a sequence of as2_msgs__srv__SetOrigin_Request.
typedef struct as2_msgs__srv__SetOrigin_Request__Sequence
{
  as2_msgs__srv__SetOrigin_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetOrigin_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetOrigin in the package as2_msgs.
typedef struct as2_msgs__srv__SetOrigin_Response
{
  /// whether the origin has been set or not
  bool success;
} as2_msgs__srv__SetOrigin_Response;

// Struct for a sequence of as2_msgs__srv__SetOrigin_Response.
typedef struct as2_msgs__srv__SetOrigin_Response__Sequence
{
  as2_msgs__srv__SetOrigin_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetOrigin_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__SET_ORIGIN__STRUCT_H_
