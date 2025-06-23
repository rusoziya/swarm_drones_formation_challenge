// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/GetGeozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__GET_GEOZONE__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__GET_GEOZONE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetGeozone in the package as2_msgs.
typedef struct as2_msgs__srv__GetGeozone_Request
{
  uint8_t structure_needs_at_least_one_member;
} as2_msgs__srv__GetGeozone_Request;

// Struct for a sequence of as2_msgs__srv__GetGeozone_Request.
typedef struct as2_msgs__srv__GetGeozone_Request__Sequence
{
  as2_msgs__srv__GetGeozone_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__GetGeozone_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'geozone_list'
#include "as2_msgs/msg/detail/geozone__struct.h"

/// Struct defined in srv/GetGeozone in the package as2_msgs.
typedef struct as2_msgs__srv__GetGeozone_Response
{
  /// whether the geofence has been set or not
  bool success;
  /// geofences stored in memory
  as2_msgs__msg__Geozone__Sequence geozone_list;
} as2_msgs__srv__GetGeozone_Response;

// Struct for a sequence of as2_msgs__srv__GetGeozone_Response.
typedef struct as2_msgs__srv__GetGeozone_Response__Sequence
{
  as2_msgs__srv__GetGeozone_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__GetGeozone_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__GET_GEOZONE__STRUCT_H_
