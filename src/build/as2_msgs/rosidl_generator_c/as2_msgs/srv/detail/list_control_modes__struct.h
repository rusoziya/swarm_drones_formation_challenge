// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/ListControlModes.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/ListControlModes in the package as2_msgs.
typedef struct as2_msgs__srv__ListControlModes_Request
{
  uint8_t structure_needs_at_least_one_member;
} as2_msgs__srv__ListControlModes_Request;

// Struct for a sequence of as2_msgs__srv__ListControlModes_Request.
typedef struct as2_msgs__srv__ListControlModes_Request__Sequence
{
  as2_msgs__srv__ListControlModes_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__ListControlModes_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'source'
#include "rosidl_runtime_c/string.h"
// Member 'control_modes'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/ListControlModes in the package as2_msgs.
typedef struct as2_msgs__srv__ListControlModes_Response
{
  /// Control modes source
  rosidl_runtime_c__String source;
  /// Control modes list
  rosidl_runtime_c__uint8__Sequence control_modes;
} as2_msgs__srv__ListControlModes_Response;

// Struct for a sequence of as2_msgs__srv__ListControlModes_Response.
typedef struct as2_msgs__srv__ListControlModes_Response__Sequence
{
  as2_msgs__srv__ListControlModes_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__ListControlModes_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__STRUCT_H_
