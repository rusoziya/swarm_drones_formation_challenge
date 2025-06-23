// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from as2_msgs:srv/ListControlModes.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "as2_msgs/srv/detail/list_control_modes__rosidl_typesupport_introspection_c.h"
#include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "as2_msgs/srv/detail/list_control_modes__functions.h"
#include "as2_msgs/srv/detail/list_control_modes__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  as2_msgs__srv__ListControlModes_Request__init(message_memory);
}

void as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_fini_function(void * message_memory)
{
  as2_msgs__srv__ListControlModes_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__srv__ListControlModes_Request, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_members = {
  "as2_msgs__srv",  // message namespace
  "ListControlModes_Request",  // message name
  1,  // number of fields
  sizeof(as2_msgs__srv__ListControlModes_Request),
  as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_member_array,  // message members
  as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_type_support_handle = {
  0,
  &as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes_Request)() {
  if (!as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_type_support_handle.typesupport_identifier) {
    as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &as2_msgs__srv__ListControlModes_Request__rosidl_typesupport_introspection_c__ListControlModes_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "as2_msgs/srv/detail/list_control_modes__rosidl_typesupport_introspection_c.h"
// already included above
// #include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "as2_msgs/srv/detail/list_control_modes__functions.h"
// already included above
// #include "as2_msgs/srv/detail/list_control_modes__struct.h"


// Include directives for member types
// Member `source`
#include "rosidl_runtime_c/string_functions.h"
// Member `control_modes`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  as2_msgs__srv__ListControlModes_Response__init(message_memory);
}

void as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_fini_function(void * message_memory)
{
  as2_msgs__srv__ListControlModes_Response__fini(message_memory);
}

size_t as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__size_function__ListControlModes_Response__control_modes(
  const void * untyped_member)
{
  const rosidl_runtime_c__uint8__Sequence * member =
    (const rosidl_runtime_c__uint8__Sequence *)(untyped_member);
  return member->size;
}

const void * as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__get_const_function__ListControlModes_Response__control_modes(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__uint8__Sequence * member =
    (const rosidl_runtime_c__uint8__Sequence *)(untyped_member);
  return &member->data[index];
}

void * as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__get_function__ListControlModes_Response__control_modes(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__uint8__Sequence * member =
    (rosidl_runtime_c__uint8__Sequence *)(untyped_member);
  return &member->data[index];
}

void as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__fetch_function__ListControlModes_Response__control_modes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__get_const_function__ListControlModes_Response__control_modes(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__assign_function__ListControlModes_Response__control_modes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__get_function__ListControlModes_Response__control_modes(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

bool as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__resize_function__ListControlModes_Response__control_modes(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__uint8__Sequence * member =
    (rosidl_runtime_c__uint8__Sequence *)(untyped_member);
  rosidl_runtime_c__uint8__Sequence__fini(member);
  return rosidl_runtime_c__uint8__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_member_array[2] = {
  {
    "source",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__srv__ListControlModes_Response, source),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "control_modes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__srv__ListControlModes_Response, control_modes),  // bytes offset in struct
    NULL,  // default value
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__size_function__ListControlModes_Response__control_modes,  // size() function pointer
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__get_const_function__ListControlModes_Response__control_modes,  // get_const(index) function pointer
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__get_function__ListControlModes_Response__control_modes,  // get(index) function pointer
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__fetch_function__ListControlModes_Response__control_modes,  // fetch(index, &value) function pointer
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__assign_function__ListControlModes_Response__control_modes,  // assign(index, value) function pointer
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__resize_function__ListControlModes_Response__control_modes  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_members = {
  "as2_msgs__srv",  // message namespace
  "ListControlModes_Response",  // message name
  2,  // number of fields
  sizeof(as2_msgs__srv__ListControlModes_Response),
  as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_member_array,  // message members
  as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_type_support_handle = {
  0,
  &as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes_Response)() {
  if (!as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_type_support_handle.typesupport_identifier) {
    as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &as2_msgs__srv__ListControlModes_Response__rosidl_typesupport_introspection_c__ListControlModes_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "as2_msgs/srv/detail/list_control_modes__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_members = {
  "as2_msgs__srv",  // service namespace
  "ListControlModes",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_Request_message_type_support_handle,
  NULL  // response message
  // as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_Response_message_type_support_handle
};

static rosidl_service_type_support_t as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_type_support_handle = {
  0,
  &as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes)() {
  if (!as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_type_support_handle.typesupport_identifier) {
    as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, srv, ListControlModes_Response)()->data;
  }

  return &as2_msgs__srv__detail__list_control_modes__rosidl_typesupport_introspection_c__ListControlModes_service_type_support_handle;
}
