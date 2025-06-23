// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/TrajGenInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/traj_gen_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'node_status'
#include "as2_msgs/msg/detail/node_status__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const TrajGenInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: node_status
  {
    out << "node_status: ";
    to_flow_style_yaml(msg.node_status, out);
    out << ", ";
  }

  // member: active_status
  {
    out << "active_status: ";
    rosidl_generator_traits::value_to_yaml(msg.active_status, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const TrajGenInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: node_status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "node_status:\n";
    to_block_style_yaml(msg.node_status, out, indentation + 2);
  }

  // member: active_status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "active_status: ";
    rosidl_generator_traits::value_to_yaml(msg.active_status, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TrajGenInfo & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::msg::TrajGenInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::TrajGenInfo & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::TrajGenInfo>()
{
  return "as2_msgs::msg::TrajGenInfo";
}

template<>
inline const char * name<as2_msgs::msg::TrajGenInfo>()
{
  return "as2_msgs/msg/TrajGenInfo";
}

template<>
struct has_fixed_size<as2_msgs::msg::TrajGenInfo>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::msg::NodeStatus>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<as2_msgs::msg::TrajGenInfo>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::msg::NodeStatus>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<as2_msgs::msg::TrajGenInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__TRAITS_HPP_
