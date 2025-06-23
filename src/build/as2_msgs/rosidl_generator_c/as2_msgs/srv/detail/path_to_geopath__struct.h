// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/PathToGeopath.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__PATH_TO_GEOPATH__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__PATH_TO_GEOPATH__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'path'
#include "nav_msgs/msg/detail/path__struct.h"

/// Struct defined in srv/PathToGeopath in the package as2_msgs.
typedef struct as2_msgs__srv__PathToGeopath_Request
{
  /// Path (m)
  nav_msgs__msg__Path path;
} as2_msgs__srv__PathToGeopath_Request;

// Struct for a sequence of as2_msgs__srv__PathToGeopath_Request.
typedef struct as2_msgs__srv__PathToGeopath_Request__Sequence
{
  as2_msgs__srv__PathToGeopath_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__PathToGeopath_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'geo_path'
#include "geographic_msgs/msg/detail/geo_path__struct.h"

/// Struct defined in srv/PathToGeopath in the package as2_msgs.
typedef struct as2_msgs__srv__PathToGeopath_Response
{
  /// whether the origin has been set or not
  bool success;
  /// Path in lat/lon and altitude
  geographic_msgs__msg__GeoPath geo_path;
} as2_msgs__srv__PathToGeopath_Response;

// Struct for a sequence of as2_msgs__srv__PathToGeopath_Response.
typedef struct as2_msgs__srv__PathToGeopath_Response__Sequence
{
  as2_msgs__srv__PathToGeopath_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__PathToGeopath_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__PATH_TO_GEOPATH__STRUCT_H_
