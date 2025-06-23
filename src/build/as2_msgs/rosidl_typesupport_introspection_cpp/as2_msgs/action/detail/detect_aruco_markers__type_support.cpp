// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from as2_msgs:action/DetectArucoMarkers.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace as2_msgs
{

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_Goal_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_Goal(_init);
}

void DetectArucoMarkers_Goal_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_Goal *>(message_memory);
  typed_message->~DetectArucoMarkers_Goal();
}

size_t size_function__DetectArucoMarkers_Goal__target_ids(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<uint16_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__DetectArucoMarkers_Goal__target_ids(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<uint16_t> *>(untyped_member);
  return &member[index];
}

void * get_function__DetectArucoMarkers_Goal__target_ids(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<uint16_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__DetectArucoMarkers_Goal__target_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint16_t *>(
    get_const_function__DetectArucoMarkers_Goal__target_ids(untyped_member, index));
  auto & value = *reinterpret_cast<uint16_t *>(untyped_value);
  value = item;
}

void assign_function__DetectArucoMarkers_Goal__target_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint16_t *>(
    get_function__DetectArucoMarkers_Goal__target_ids(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint16_t *>(untyped_value);
  item = value;
}

void resize_function__DetectArucoMarkers_Goal__target_ids(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<uint16_t> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_Goal_message_member_array[1] = {
  {
    "target_ids",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_Goal, target_ids),  // bytes offset in struct
    nullptr,  // default value
    size_function__DetectArucoMarkers_Goal__target_ids,  // size() function pointer
    get_const_function__DetectArucoMarkers_Goal__target_ids,  // get_const(index) function pointer
    get_function__DetectArucoMarkers_Goal__target_ids,  // get(index) function pointer
    fetch_function__DetectArucoMarkers_Goal__target_ids,  // fetch(index, &value) function pointer
    assign_function__DetectArucoMarkers_Goal__target_ids,  // assign(index, value) function pointer
    resize_function__DetectArucoMarkers_Goal__target_ids  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_Goal_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_Goal",  // message name
  1,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_Goal),
  DetectArucoMarkers_Goal_message_member_array,  // message members
  DetectArucoMarkers_Goal_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_Goal_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_Goal_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_Goal_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_Goal>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_Goal_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_Goal)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_Goal_message_type_support_handle;
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_Result_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_Result(_init);
}

void DetectArucoMarkers_Result_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_Result *>(message_memory);
  typed_message->~DetectArucoMarkers_Result();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_Result_message_member_array[1] = {
  {
    "success",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_Result, success),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_Result_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_Result",  // message name
  1,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_Result),
  DetectArucoMarkers_Result_message_member_array,  // message members
  DetectArucoMarkers_Result_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_Result_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_Result_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_Result_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_Result>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_Result_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_Result)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_Result_message_type_support_handle;
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_Feedback_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_Feedback(_init);
}

void DetectArucoMarkers_Feedback_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_Feedback *>(message_memory);
  typed_message->~DetectArucoMarkers_Feedback();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_Feedback_message_member_array[1] = {
  {
    "sucess",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_Feedback, sucess),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_Feedback_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_Feedback",  // message name
  1,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_Feedback),
  DetectArucoMarkers_Feedback_message_member_array,  // message members
  DetectArucoMarkers_Feedback_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_Feedback_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_Feedback_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_Feedback_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_Feedback>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_Feedback_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_Feedback)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_Feedback_message_type_support_handle;
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_SendGoal_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_SendGoal_Request(_init);
}

void DetectArucoMarkers_SendGoal_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_SendGoal_Request *>(message_memory);
  typed_message->~DetectArucoMarkers_SendGoal_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_SendGoal_Request_message_member_array[2] = {
  {
    "goal_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<unique_identifier_msgs::msg::UUID>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_SendGoal_Request, goal_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "goal",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_Goal>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_SendGoal_Request, goal),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_SendGoal_Request_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_SendGoal_Request",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_SendGoal_Request),
  DetectArucoMarkers_SendGoal_Request_message_member_array,  // message members
  DetectArucoMarkers_SendGoal_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_SendGoal_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_SendGoal_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_SendGoal_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_SendGoal_Request>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_SendGoal_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_SendGoal_Request)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_SendGoal_Request_message_type_support_handle;
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_SendGoal_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_SendGoal_Response(_init);
}

void DetectArucoMarkers_SendGoal_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_SendGoal_Response *>(message_memory);
  typed_message->~DetectArucoMarkers_SendGoal_Response();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_SendGoal_Response_message_member_array[2] = {
  {
    "accepted",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_SendGoal_Response, accepted),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "stamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_SendGoal_Response, stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_SendGoal_Response_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_SendGoal_Response",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_SendGoal_Response),
  DetectArucoMarkers_SendGoal_Response_message_member_array,  // message members
  DetectArucoMarkers_SendGoal_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_SendGoal_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_SendGoal_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_SendGoal_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_SendGoal_Response>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_SendGoal_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_SendGoal_Response)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_SendGoal_Response_message_type_support_handle;
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace as2_msgs
{

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers DetectArucoMarkers_SendGoal_service_members = {
  "as2_msgs::action",  // service namespace
  "DetectArucoMarkers_SendGoal",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<as2_msgs::action::DetectArucoMarkers_SendGoal>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t DetectArucoMarkers_SendGoal_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_SendGoal_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<as2_msgs::action::DetectArucoMarkers_SendGoal>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_SendGoal_service_type_support_handle;
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
        ::as2_msgs::action::DetectArucoMarkers_SendGoal_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::as2_msgs::action::DetectArucoMarkers_SendGoal_Response
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
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_SendGoal)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<as2_msgs::action::DetectArucoMarkers_SendGoal>();
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_GetResult_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_GetResult_Request(_init);
}

void DetectArucoMarkers_GetResult_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_GetResult_Request *>(message_memory);
  typed_message->~DetectArucoMarkers_GetResult_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_GetResult_Request_message_member_array[1] = {
  {
    "goal_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<unique_identifier_msgs::msg::UUID>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_GetResult_Request, goal_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_GetResult_Request_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_GetResult_Request",  // message name
  1,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_GetResult_Request),
  DetectArucoMarkers_GetResult_Request_message_member_array,  // message members
  DetectArucoMarkers_GetResult_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_GetResult_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_GetResult_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_GetResult_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_GetResult_Request>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_GetResult_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_GetResult_Request)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_GetResult_Request_message_type_support_handle;
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_GetResult_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_GetResult_Response(_init);
}

void DetectArucoMarkers_GetResult_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_GetResult_Response *>(message_memory);
  typed_message->~DetectArucoMarkers_GetResult_Response();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_GetResult_Response_message_member_array[2] = {
  {
    "status",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_GetResult_Response, status),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "result",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_Result>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_GetResult_Response, result),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_GetResult_Response_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_GetResult_Response",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_GetResult_Response),
  DetectArucoMarkers_GetResult_Response_message_member_array,  // message members
  DetectArucoMarkers_GetResult_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_GetResult_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_GetResult_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_GetResult_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_GetResult_Response>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_GetResult_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_GetResult_Response)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_GetResult_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace as2_msgs
{

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers DetectArucoMarkers_GetResult_service_members = {
  "as2_msgs::action",  // service namespace
  "DetectArucoMarkers_GetResult",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<as2_msgs::action::DetectArucoMarkers_GetResult>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t DetectArucoMarkers_GetResult_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_GetResult_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<as2_msgs::action::DetectArucoMarkers_GetResult>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_GetResult_service_type_support_handle;
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
        ::as2_msgs::action::DetectArucoMarkers_GetResult_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::as2_msgs::action::DetectArucoMarkers_GetResult_Response
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
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_GetResult)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<as2_msgs::action::DetectArucoMarkers_GetResult>();
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
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
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

namespace action
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectArucoMarkers_FeedbackMessage_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::DetectArucoMarkers_FeedbackMessage(_init);
}

void DetectArucoMarkers_FeedbackMessage_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::DetectArucoMarkers_FeedbackMessage *>(message_memory);
  typed_message->~DetectArucoMarkers_FeedbackMessage();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectArucoMarkers_FeedbackMessage_message_member_array[2] = {
  {
    "goal_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<unique_identifier_msgs::msg::UUID>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_FeedbackMessage, goal_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "feedback",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_Feedback>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::DetectArucoMarkers_FeedbackMessage, feedback),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectArucoMarkers_FeedbackMessage_message_members = {
  "as2_msgs::action",  // message namespace
  "DetectArucoMarkers_FeedbackMessage",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::DetectArucoMarkers_FeedbackMessage),
  DetectArucoMarkers_FeedbackMessage_message_member_array,  // message members
  DetectArucoMarkers_FeedbackMessage_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectArucoMarkers_FeedbackMessage_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectArucoMarkers_FeedbackMessage_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectArucoMarkers_FeedbackMessage_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace action

}  // namespace as2_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::action::DetectArucoMarkers_FeedbackMessage>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_FeedbackMessage_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, DetectArucoMarkers_FeedbackMessage)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::DetectArucoMarkers_FeedbackMessage_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
