// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from as2_msgs:msg/PolygonList.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "as2_msgs/msg/detail/polygon_list__struct.hpp"
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

void PolygonList_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) as2_msgs::msg::PolygonList(_init);
}

void PolygonList_fini_function(void * message_memory)
{
  auto typed_message = static_cast<as2_msgs::msg::PolygonList *>(message_memory);
  typed_message->~PolygonList();
}

size_t size_function__PolygonList__polygons(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::PolygonStamped> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PolygonList__polygons(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::PolygonStamped> *>(untyped_member);
  return &member[index];
}

void * get_function__PolygonList__polygons(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::PolygonStamped> *>(untyped_member);
  return &member[index];
}

void fetch_function__PolygonList__polygons(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const geometry_msgs::msg::PolygonStamped *>(
    get_const_function__PolygonList__polygons(untyped_member, index));
  auto & value = *reinterpret_cast<geometry_msgs::msg::PolygonStamped *>(untyped_value);
  value = item;
}

void assign_function__PolygonList__polygons(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<geometry_msgs::msg::PolygonStamped *>(
    get_function__PolygonList__polygons(untyped_member, index));
  const auto & value = *reinterpret_cast<const geometry_msgs::msg::PolygonStamped *>(untyped_value);
  item = value;
}

void resize_function__PolygonList__polygons(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::PolygonStamped> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PolygonList_message_member_array[1] = {
  {
    "polygons",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::PolygonStamped>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs::msg::PolygonList, polygons),  // bytes offset in struct
    nullptr,  // default value
    size_function__PolygonList__polygons,  // size() function pointer
    get_const_function__PolygonList__polygons,  // get_const(index) function pointer
    get_function__PolygonList__polygons,  // get(index) function pointer
    fetch_function__PolygonList__polygons,  // fetch(index, &value) function pointer
    assign_function__PolygonList__polygons,  // assign(index, value) function pointer
    resize_function__PolygonList__polygons  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PolygonList_message_members = {
  "as2_msgs::msg",  // message namespace
  "PolygonList",  // message name
  1,  // number of fields
  sizeof(as2_msgs::msg::PolygonList),
  PolygonList_message_member_array,  // message members
  PolygonList_init_function,  // function to initialize message memory (memory has to be allocated)
  PolygonList_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PolygonList_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PolygonList_message_members,
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
get_message_type_support_handle<as2_msgs::msg::PolygonList>()
{
  return &::as2_msgs::msg::rosidl_typesupport_introspection_cpp::PolygonList_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, msg, PolygonList)() {
  return &::as2_msgs::msg::rosidl_typesupport_introspection_cpp::PolygonList_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
