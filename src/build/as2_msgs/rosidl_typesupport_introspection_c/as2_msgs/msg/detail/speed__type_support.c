// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from as2_msgs:msg/Speed.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "as2_msgs/msg/detail/speed__rosidl_typesupport_introspection_c.h"
#include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "as2_msgs/msg/detail/speed__functions.h"
#include "as2_msgs/msg/detail/speed__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  as2_msgs__msg__Speed__init(message_memory);
}

void as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_fini_function(void * message_memory)
{
  as2_msgs__msg__Speed__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__Speed, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__Speed, speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_members = {
  "as2_msgs__msg",  // message namespace
  "Speed",  // message name
  2,  // number of fields
  sizeof(as2_msgs__msg__Speed),
  as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_member_array,  // message members
  as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_init_function,  // function to initialize message memory (memory has to be allocated)
  as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_type_support_handle = {
  0,
  &as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, msg, Speed)() {
  as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_type_support_handle.typesupport_identifier) {
    as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &as2_msgs__msg__Speed__rosidl_typesupport_introspection_c__Speed_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
