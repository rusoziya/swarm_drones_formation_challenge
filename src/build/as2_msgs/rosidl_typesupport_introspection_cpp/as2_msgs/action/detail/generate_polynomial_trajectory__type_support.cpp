// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from as2_msgs:action/GeneratePolynomialTrajectory.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_Goal_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_Goal(_init);
}

void GeneratePolynomialTrajectory_Goal_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_Goal *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_Goal();
}

size_t size_function__GeneratePolynomialTrajectory_Goal__path(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<as2_msgs::msg::PoseWithID> *>(untyped_member);
  return member->size();
}

const void * get_const_function__GeneratePolynomialTrajectory_Goal__path(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<as2_msgs::msg::PoseWithID> *>(untyped_member);
  return &member[index];
}

void * get_function__GeneratePolynomialTrajectory_Goal__path(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<as2_msgs::msg::PoseWithID> *>(untyped_member);
  return &member[index];
}

void fetch_function__GeneratePolynomialTrajectory_Goal__path(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const as2_msgs::msg::PoseWithID *>(
    get_const_function__GeneratePolynomialTrajectory_Goal__path(untyped_member, index));
  auto & value = *reinterpret_cast<as2_msgs::msg::PoseWithID *>(untyped_value);
  value = item;
}

void assign_function__GeneratePolynomialTrajectory_Goal__path(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<as2_msgs::msg::PoseWithID *>(
    get_function__GeneratePolynomialTrajectory_Goal__path(untyped_member, index));
  const auto & value = *reinterpret_cast<const as2_msgs::msg::PoseWithID *>(untyped_value);
  item = value;
}

void resize_function__GeneratePolynomialTrajectory_Goal__path(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<as2_msgs::msg::PoseWithID> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_Goal_message_member_array[4] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Goal, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "yaw",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::msg::YawMode>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Goal, yaw),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "path",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::msg::PoseWithID>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Goal, path),  // bytes offset in struct
    nullptr,  // default value
    size_function__GeneratePolynomialTrajectory_Goal__path,  // size() function pointer
    get_const_function__GeneratePolynomialTrajectory_Goal__path,  // get_const(index) function pointer
    get_function__GeneratePolynomialTrajectory_Goal__path,  // get(index) function pointer
    fetch_function__GeneratePolynomialTrajectory_Goal__path,  // fetch(index, &value) function pointer
    assign_function__GeneratePolynomialTrajectory_Goal__path,  // assign(index, value) function pointer
    resize_function__GeneratePolynomialTrajectory_Goal__path  // resize(index) function pointer
  },
  {
    "max_speed",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Goal, max_speed),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_Goal_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_Goal",  // message name
  4,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_Goal),
  GeneratePolynomialTrajectory_Goal_message_member_array,  // message members
  GeneratePolynomialTrajectory_Goal_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_Goal_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_Goal_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_Goal_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_Goal>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_Goal_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_Goal)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_Goal_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_Result_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_Result(_init);
}

void GeneratePolynomialTrajectory_Result_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_Result *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_Result();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_Result_message_member_array[1] = {
  {
    "trajectory_generator_success",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Result, trajectory_generator_success),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_Result_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_Result",  // message name
  1,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_Result),
  GeneratePolynomialTrajectory_Result_message_member_array,  // message members
  GeneratePolynomialTrajectory_Result_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_Result_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_Result_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_Result_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_Result>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_Result_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_Result)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_Result_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_Feedback_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_Feedback(_init);
}

void GeneratePolynomialTrajectory_Feedback_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_Feedback *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_Feedback();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_Feedback_message_member_array[2] = {
  {
    "next_waypoint_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Feedback, next_waypoint_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "remaining_waypoints",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_Feedback, remaining_waypoints),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_Feedback_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_Feedback",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_Feedback),
  GeneratePolynomialTrajectory_Feedback_message_member_array,  // message members
  GeneratePolynomialTrajectory_Feedback_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_Feedback_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_Feedback_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_Feedback_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_Feedback>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_Feedback_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_Feedback)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_Feedback_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_SendGoal_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request(_init);
}

void GeneratePolynomialTrajectory_SendGoal_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_SendGoal_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_SendGoal_Request_message_member_array[2] = {
  {
    "goal_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<unique_identifier_msgs::msg::UUID>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request, goal_id),  // bytes offset in struct
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
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_Goal>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request, goal),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_SendGoal_Request_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_SendGoal_Request",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request),
  GeneratePolynomialTrajectory_SendGoal_Request_message_member_array,  // message members
  GeneratePolynomialTrajectory_SendGoal_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_SendGoal_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_SendGoal_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_SendGoal_Request_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_SendGoal_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_SendGoal_Request)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_SendGoal_Request_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_SendGoal_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response(_init);
}

void GeneratePolynomialTrajectory_SendGoal_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_SendGoal_Response();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_SendGoal_Response_message_member_array[2] = {
  {
    "accepted",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response, accepted),  // bytes offset in struct
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
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response, stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_SendGoal_Response_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_SendGoal_Response",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response),
  GeneratePolynomialTrajectory_SendGoal_Response_message_member_array,  // message members
  GeneratePolynomialTrajectory_SendGoal_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_SendGoal_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_SendGoal_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_SendGoal_Response_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_SendGoal_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_SendGoal_Response)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_SendGoal_Response_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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
static ::rosidl_typesupport_introspection_cpp::ServiceMembers GeneratePolynomialTrajectory_SendGoal_service_members = {
  "as2_msgs::action",  // service namespace
  "GeneratePolynomialTrajectory_SendGoal",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t GeneratePolynomialTrajectory_SendGoal_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_SendGoal_service_members,
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
get_service_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_SendGoal_service_type_support_handle;
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
        ::as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::as2_msgs::action::GeneratePolynomialTrajectory_SendGoal_Response
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
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_SendGoal)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_SendGoal>();
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_GetResult_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Request(_init);
}

void GeneratePolynomialTrajectory_GetResult_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Request *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_GetResult_Request();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_GetResult_Request_message_member_array[1] = {
  {
    "goal_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<unique_identifier_msgs::msg::UUID>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Request, goal_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_GetResult_Request_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_GetResult_Request",  // message name
  1,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Request),
  GeneratePolynomialTrajectory_GetResult_Request_message_member_array,  // message members
  GeneratePolynomialTrajectory_GetResult_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_GetResult_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_GetResult_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_GetResult_Request_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Request>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_GetResult_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_GetResult_Request)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_GetResult_Request_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_GetResult_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response(_init);
}

void GeneratePolynomialTrajectory_GetResult_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_GetResult_Response();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_GetResult_Response_message_member_array[2] = {
  {
    "status",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response, status),  // bytes offset in struct
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
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_Result>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response, result),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_GetResult_Response_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_GetResult_Response",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response),
  GeneratePolynomialTrajectory_GetResult_Response_message_member_array,  // message members
  GeneratePolynomialTrajectory_GetResult_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_GetResult_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_GetResult_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_GetResult_Response_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_GetResult_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_GetResult_Response)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_GetResult_Response_message_type_support_handle;
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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
static ::rosidl_typesupport_introspection_cpp::ServiceMembers GeneratePolynomialTrajectory_GetResult_service_members = {
  "as2_msgs::action",  // service namespace
  "GeneratePolynomialTrajectory_GetResult",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_GetResult>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t GeneratePolynomialTrajectory_GetResult_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_GetResult_service_members,
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
get_service_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_GetResult>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_GetResult_service_type_support_handle;
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
        ::as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::as2_msgs::action::GeneratePolynomialTrajectory_GetResult_Response
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
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_GetResult)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_GetResult>();
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
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.hpp"
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

void GeneratePolynomialTrajectory_FeedbackMessage_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::action::GeneratePolynomialTrajectory_FeedbackMessage(_init);
}

void GeneratePolynomialTrajectory_FeedbackMessage_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::action::GeneratePolynomialTrajectory_FeedbackMessage *>(message_memory);
  typed_message->~GeneratePolynomialTrajectory_FeedbackMessage();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GeneratePolynomialTrajectory_FeedbackMessage_message_member_array[2] = {
  {
    "goal_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<unique_identifier_msgs::msg::UUID>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_FeedbackMessage, goal_id),  // bytes offset in struct
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
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_Feedback>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::action::GeneratePolynomialTrajectory_FeedbackMessage, feedback),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GeneratePolynomialTrajectory_FeedbackMessage_message_members = {
  "as2_msgs::action",  // message namespace
  "GeneratePolynomialTrajectory_FeedbackMessage",  // message name
  2,  // number of fields
  sizeof(as2_msgs::action::GeneratePolynomialTrajectory_FeedbackMessage),
  GeneratePolynomialTrajectory_FeedbackMessage_message_member_array,  // message members
  GeneratePolynomialTrajectory_FeedbackMessage_init_function,  // function to initialize message memory (memory has to be allocated)
  GeneratePolynomialTrajectory_FeedbackMessage_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GeneratePolynomialTrajectory_FeedbackMessage_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GeneratePolynomialTrajectory_FeedbackMessage_message_members,
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
get_message_type_support_handle<as2_msgs::action::GeneratePolynomialTrajectory_FeedbackMessage>()
{
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_FeedbackMessage_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, action, GeneratePolynomialTrajectory_FeedbackMessage)() {
  return &::as2_msgs::action::rosidl_typesupport_introspection_cpp::GeneratePolynomialTrajectory_FeedbackMessage_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
