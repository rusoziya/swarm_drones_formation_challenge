// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/PlatformInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/platform_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'status'
#include "as2_msgs/msg/detail/platform_status__traits.hpp"
// Member 'current_control_mode'
#include "as2_msgs/msg/detail/control_mode__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PlatformInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: connected
  {
    out << "connected: ";
    rosidl_generator_traits::value_to_yaml(msg.connected, out);
    out << ", ";
  }

  // member: armed
  {
    out << "armed: ";
    rosidl_generator_traits::value_to_yaml(msg.armed, out);
    out << ", ";
  }

  // member: offboard
  {
    out << "offboard: ";
    rosidl_generator_traits::value_to_yaml(msg.offboard, out);
    out << ", ";
  }

  // member: status
  {
    out << "status: ";
    to_flow_style_yaml(msg.status, out);
    out << ", ";
  }

  // member: current_control_mode
  {
    out << "current_control_mode: ";
    to_flow_style_yaml(msg.current_control_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PlatformInfo & msg,
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

  // member: connected
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "connected: ";
    rosidl_generator_traits::value_to_yaml(msg.connected, out);
    out << "\n";
  }

  // member: armed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "armed: ";
    rosidl_generator_traits::value_to_yaml(msg.armed, out);
    out << "\n";
  }

  // member: offboard
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "offboard: ";
    rosidl_generator_traits::value_to_yaml(msg.offboard, out);
    out << "\n";
  }

  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status:\n";
    to_block_style_yaml(msg.status, out, indentation + 2);
  }

  // member: current_control_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_control_mode:\n";
    to_block_style_yaml(msg.current_control_mode, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PlatformInfo & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::PlatformInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::PlatformInfo & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::PlatformInfo>()
{
  return "as2_msgs::msg::PlatformInfo";
}

template<>
inline const char * name<as2_msgs::msg::PlatformInfo>()
{
  return "as2_msgs/msg/PlatformInfo";
}

template<>
struct has_fixed_size<as2_msgs::msg::PlatformInfo>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::msg::ControlMode>::value && has_fixed_size<as2_msgs::msg::PlatformStatus>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<as2_msgs::msg::PlatformInfo>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::msg::ControlMode>::value && has_bounded_size<as2_msgs::msg::PlatformStatus>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<as2_msgs::msg::PlatformInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__TRAITS_HPP_
