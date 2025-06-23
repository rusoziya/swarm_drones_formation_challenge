// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from as2_msgs:srv/ListControlModes.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "as2_msgs/srv/detail/list_control_modes__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace as2_msgs
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void ListControlModes_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::srv::ListControlModes_Request(_init);
}

void ListControlModes_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::srv::ListControlModes_Request *>(message_memory);
  typed_message->~ListControlModes_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ListControlModes_Request_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::srv::ListControlModes_Request, structure_needs_at_least_one_member),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ListControlModes_Request_message_members = {
  "as2_msgs::srv",  // message namespace
  "ListControlModes_Request",  // message name
  1,  // number of fields
  sizeof(as2_msgs::srv::ListControlModes_Request),
  ListControlModes_Request_message_member_array,  // message members
  ListControlModes_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  ListControlModes_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ListControlModes_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ListControlModes_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::srv::ListControlModes_Request>()
{
  return &::as2_msgs::srv::rosidl_typesupport_introspection_cpp::ListControlModes_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, srv, ListControlModes_Request)() {
  return &::as2_msgs::srv::rosidl_typesupport_introspection_cpp::ListControlModes_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "array"
// already included above
// #include "cstddef"
// already included above
// #include "string"
// already included above
// #include "vector"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "as2_msgs/srv/detail/list_control_modes__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/field_types.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace as2_msgs
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void ListControlModes_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::srv::ListControlModes_Response(_init);
}

void ListControlModes_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::srv::ListControlModes_Response *>(message_memory);
  typed_message->~ListControlModes_Response();
}

size_t size_function__ListControlModes_Response__control_modes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<uint8_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ListControlModes_Response__control_modes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<uint8_t> *>(untyped_member);
  return &member[index];
}

void * get_function__ListControlModes_Response__control_modes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<uint8_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__ListControlModes_Response__control_modes(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__ListControlModes_Response__control_modes(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__ListControlModes_Response__control_modes(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__ListControlModes_Response__control_modes(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

void resize_function__ListControlModes_Response__control_modes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<uint8_t> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ListControlModes_Response_message_member_array[2] = {
  {
    "source",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::srv::ListControlModes_Response, source),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "control_modes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::srv::ListControlModes_Response, control_modes),  // bytes offset in struct
    nullptr,  // default value
    size_function__ListControlModes_Response__control_modes,  // size() function pointer
    get_const_function__ListControlModes_Response__control_modes,  // get_const(index) function pointer
    get_function__ListControlModes_Response__control_modes,  // get(index) function pointer
    fetch_function__ListControlModes_Response__control_modes,  // fetch(index, &value) function pointer
    assign_function__ListControlModes_Response__control_modes,  // assign(index, value) function pointer
    resize_function__ListControlModes_Response__control_modes  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ListControlModes_Response_message_members = {
  "as2_msgs::srv",  // message namespace
  "ListControlModes_Response",  // message name
  2,  // number of fields
  sizeof(as2_msgs::srv::ListControlModes_Response),
  ListControlModes_Response_message_member_array,  // message members
  ListControlModes_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  ListControlModes_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ListControlModes_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ListControlModes_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::srv::ListControlModes_Response>()
{
  return &::as2_msgs::srv::rosidl_typesupport_introspection_cpp::ListControlModes_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, srv, ListControlModes_Response)() {
  return &::as2_msgs::srv::rosidl_typesupport_introspection_cpp::ListControlModes_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"
// already included above
// #include "as2_msgs/srv/detail/list_control_modes__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace as2_msgs
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers ListControlModes_service_members = {
  "as2_msgs::srv",  // service namespace
  "ListControlModes",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<as2_msgs::srv::ListControlModes>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t ListControlModes_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ListControlModes_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<as2_msgs::srv::ListControlModes>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::as2_msgs::srv::rosidl_typesupport_introspection_cpp::ListControlModes_service_type_support_handle;
  // get a non-const and properly typed version of the data void *
  auto service_members = const_cast<::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
    static_cast<const ::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
      service_type_support->data));
  // make sure that both the request_members_ and the response_members_ are initialized
  // if they are not, initialize them
  if (
    service_members->request_members_ == nullptr ||
    service_members->response_members_ == nullptr)
  {
    // initialize the request_members_ with the static function from the external library
    service_members->request_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::as2_msgs::srv::ListControlModes_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::as2_msgs::srv::ListControlModes_Response
      >()->data
      );
  }
  // finally return the properly initialized service_type_support handle
  return service_type_support;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, srv, ListControlModes)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<as2_msgs::srv::ListControlModes>();
}

#ifdef __cplusplus
}
#endif
