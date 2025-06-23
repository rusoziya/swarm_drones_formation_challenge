// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/PolygonList.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POLYGON_LIST__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__POLYGON_LIST__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'polygons'
#include "geometry_msgs/msg/detail/polygon_stamped__struct.h"

/// Struct defined in msg/PolygonList in the package as2_msgs.
/**
  * List of polygons to visualize multiple geozones in RVIZ2
 */
typedef struct as2_msgs__msg__PolygonList
{
  geometry_msgs__msg__PolygonStamped__Sequence polygons;
} as2_msgs__msg__PolygonList;

// Struct for a sequence of as2_msgs__msg__PolygonList.
typedef struct as2_msgs__msg__PolygonList__Sequence
{
  as2_msgs__msg__PolygonList * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__PolygonList__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__POLYGON_LIST__STRUCT_H_
