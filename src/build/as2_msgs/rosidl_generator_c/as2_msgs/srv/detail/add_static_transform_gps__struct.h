// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/AddStaticTransformGps.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__STRUCT_H_

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
// Member 'gps_position'
#include "sensor_msgs/msg/detail/nav_sat_fix__struct.h"

/// Struct defined in srv/AddStaticTransformGps in the package as2_msgs.
typedef struct as2_msgs__srv__AddStaticTransformGps_Request
{
  /// Parent frame frame id
  rosidl_runtime_c__String frame_id;
  /// Child frame id
  rosidl_runtime_c__String child_frame_id;
  /// Transform
  sensor_msgs__msg__NavSatFix gps_position;
  /// yaw
  float azimuth;
  /// pitch
  float elevation;
  /// roll
  float bank;
} as2_msgs__srv__AddStaticTransformGps_Request;

// Struct for a sequence of as2_msgs__srv__AddStaticTransformGps_Request.
typedef struct as2_msgs__srv__AddStaticTransformGps_Request__Sequence
{
  as2_msgs__srv__AddStaticTransformGps_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__AddStaticTransformGps_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/AddStaticTransformGps in the package as2_msgs.
typedef struct as2_msgs__srv__AddStaticTransformGps_Response
{
  /// whether the transform has been set or not
  bool success;
} as2_msgs__srv__AddStaticTransformGps_Response;

// Struct for a sequence of as2_msgs__srv__AddStaticTransformGps_Response.
typedef struct as2_msgs__srv__AddStaticTransformGps_Response__Sequence
{
  as2_msgs__srv__AddStaticTransformGps_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__AddStaticTransformGps_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__STRUCT_H_
