// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/ControllerInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/controller_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'input_control_mode'
// Member 'output_control_mode'
#include "as2_msgs/msg/detail/control_mode__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ControllerInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: input_control_mode
  {
    out << "input_control_mode: ";
    to_flow_style_yaml(msg.input_control_mode, out);
    out << ", ";
  }

  // member: output_control_mode
  {
    out << "output_control_mode: ";
    to_flow_style_yaml(msg.output_control_mode, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ControllerInfo & msg,
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

  // member: input_control_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "input_control_mode:\n";
    to_block_style_yaml(msg.input_control_mode, out, indentation + 2);
  }

  // member: output_control_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "output_control_mode:\n";
    to_block_style_yaml(msg.output_control_mode, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ControllerInfo & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::ControllerInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::ControllerInfo & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::ControllerInfo>()
{
  return "as2_msgs::msg::ControllerInfo";
}

template<>
inline const char * name<as2_msgs::msg::ControllerInfo>()
{
  return "as2_msgs/msg/ControllerInfo";
}

template<>
struct has_fixed_size<as2_msgs::msg::ControllerInfo>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::msg::ControlMode>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<as2_msgs::msg::ControllerInfo>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::msg::ControlMode>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<as2_msgs::msg::ControllerInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__TRAITS_HPP_
