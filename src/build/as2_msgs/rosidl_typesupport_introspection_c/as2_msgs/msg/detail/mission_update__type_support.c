// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from as2_msgs:msg/MissionUpdate.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "as2_msgs/msg/detail/mission_update__rosidl_typesupport_introspection_c.h"
#include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "as2_msgs/msg/detail/mission_update__functions.h"
#include "as2_msgs/msg/detail/mission_update__struct.h"


// Include directives for member types
// Member `drone_id`
// Member `mission`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  as2_msgs__msg__MissionUpdate__init(message_memory);
}

void as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_fini_function(void * message_memory)
{
  as2_msgs__msg__MissionUpdate__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_member_array[5] = {
  {
    "drone_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__MissionUpdate, drone_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mission_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__MissionUpdate, mission_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "item_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__MissionUpdate, item_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "action",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__MissionUpdate, action),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mission",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__MissionUpdate, mission),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_members = {
  "as2_msgs__msg",  // message namespace
  "MissionUpdate",  // message name
  5,  // number of fields
  sizeof(as2_msgs__msg__MissionUpdate),
  as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_member_array,  // message members
  as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_init_function,  // function to initialize message memory (memory has to be allocated)
  as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_type_support_handle = {
  0,
  &as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, msg, MissionUpdate)() {
  if (!as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_type_support_handle.typesupport_identifier) {
    as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &as2_msgs__msg__MissionUpdate__rosidl_typesupport_introspection_c__MissionUpdate_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
