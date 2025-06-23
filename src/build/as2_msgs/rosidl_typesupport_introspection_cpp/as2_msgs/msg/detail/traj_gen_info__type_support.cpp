// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from as2_msgs:msg/TrajGenInfo.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "as2_msgs/msg/detail/traj_gen_info__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace as2_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void TrajGenInfo_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::msg::TrajGenInfo(_init);
}

void TrajGenInfo_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::msg::TrajGenInfo *>(message_memory);
  typed_message->~TrajGenInfo();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember TrajGenInfo_message_member_array[3] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::msg::TrajGenInfo, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "node_status",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::msg::NodeStatus>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::msg::TrajGenInfo, node_status),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "active_status",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::msg::TrajGenInfo, active_status),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers TrajGenInfo_message_members = {
  "as2_msgs::msg",  // message namespace
  "TrajGenInfo",  // message name
  3,  // number of fields
  sizeof(as2_msgs::msg::TrajGenInfo),
  TrajGenInfo_message_member_array,  // message members
  TrajGenInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  TrajGenInfo_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t TrajGenInfo_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &TrajGenInfo_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::msg::TrajGenInfo>()
{
  return &::as2_msgs::msg::rosidl_typesupport_introspection_cpp::TrajGenInfo_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, msg, TrajGenInfo)() {
  return &::as2_msgs::msg::rosidl_typesupport_introspection_cpp::TrajGenInfo_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
