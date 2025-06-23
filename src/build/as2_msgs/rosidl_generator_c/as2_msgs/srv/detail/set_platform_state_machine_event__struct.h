// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:srv/SetPlatformStateMachineEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__STRUCT_H_
#define AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'event'
#include "as2_msgs/msg/detail/platform_state_machine_event__struct.h"

/// Struct defined in srv/SetPlatformStateMachineEvent in the package as2_msgs.
typedef struct as2_msgs__srv__SetPlatformStateMachineEvent_Request
{
  /// event to set
  as2_msgs__msg__PlatformStateMachineEvent event;
} as2_msgs__srv__SetPlatformStateMachineEvent_Request;

// Struct for a sequence of as2_msgs__srv__SetPlatformStateMachineEvent_Request.
typedef struct as2_msgs__srv__SetPlatformStateMachineEvent_Request__Sequence
{
  as2_msgs__srv__SetPlatformStateMachineEvent_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetPlatformStateMachineEvent_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'current_state'
#include "as2_msgs/msg/detail/platform_status__struct.h"

/// Struct defined in srv/SetPlatformStateMachineEvent in the package as2_msgs.
typedef struct as2_msgs__srv__SetPlatformStateMachineEvent_Response
{
  /// whether the PSM has been set or not
  bool success;
  /// PSM result of the aircraft
  as2_msgs__msg__PlatformStatus current_state;
} as2_msgs__srv__SetPlatformStateMachineEvent_Response;

// Struct for a sequence of as2_msgs__srv__SetPlatformStateMachineEvent_Response.
typedef struct as2_msgs__srv__SetPlatformStateMachineEvent_Response__Sequence
{
  as2_msgs__srv__SetPlatformStateMachineEvent_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__srv__SetPlatformStateMachineEvent_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__STRUCT_H_
