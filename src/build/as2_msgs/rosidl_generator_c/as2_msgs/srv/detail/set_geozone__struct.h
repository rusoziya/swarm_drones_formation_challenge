// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/SetGeozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_GEOZONE__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__SET_GEOZONE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'geozone'
#include "as2_msgs/msg/detail/geozone__struct.h"

/// Struct defined in srv/SetGeozone in the package as2_msgs.
typedef struct as2_msgs__srv__SetGeozone_Request
{
  /// geostructure to set
  as2_msgs__msg__Geozone geozone;
} as2_msgs__srv__SetGeozone_Request;

// Struct for a sequence of as2_msgs__srv__SetGeozone_Request.
typedef struct as2_msgs__srv__SetGeozone_Request__Sequence
{
  as2_msgs__srv__SetGeozone_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetGeozone_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetGeozone in the package as2_msgs.
typedef struct as2_msgs__srv__SetGeozone_Response
{
  /// whether the geoStructure has been set or not
  bool success;
} as2_msgs__srv__SetGeozone_Response;

// Struct for a sequence of as2_msgs__srv__SetGeozone_Response.
typedef struct as2_msgs__srv__SetGeozone_Response__Sequence
{
  as2_msgs__srv__SetGeozone_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetGeozone_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__SET_GEOZONE__STRUCT_H_
