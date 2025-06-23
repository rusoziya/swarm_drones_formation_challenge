// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/Geozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__GEOZONE__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__GEOZONE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'type'
// Member 'data_type'
#include "rosidl_runtime_c/string.h"
// Member 'polygon'
#include "geometry_msgs/msg/detail/polygon__struct.h"

/// Struct defined in msg/Geozone in the package as2_msgs.
/**
  * GeoStructure defined by an id, alert that generates in case of event and a polygon 
  * that defines the area.
 */
typedef struct as2_msgs__msg__Geozone
{
  /// geofence id
  int8_t id;
  /// alert generated
  int8_t alert;
  /// geofence or geocage
  rosidl_runtime_c__String type;
  /// cartesian or gps
  rosidl_runtime_c__String data_type;
  /// fence polygon
  geometry_msgs__msg__Polygon polygon;
  /// height limit up
  float z_up;
  /// height limit bottom
  float z_down;
} as2_msgs__msg__Geozone;

// Struct for a sequence of as2_msgs__msg__Geozone.
typedef struct as2_msgs__msg__Geozone__Sequence
{
  as2_msgs__msg__Geozone * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__Geozone__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__GEOZONE__STRUCT_H_
