// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/GimbalControl.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/gimbal_control__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'target'
#include "geometry_msgs/msg/detail/vector3_stamped__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GimbalControl & msg,
  std::ostream & out)
{
  out << "{";
  // member: control_mode
  {
    out << "control_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.control_mode, out);
    out << ", ";
  }

  // member: target
  {
    out << "target: ";
    to_flow_style_yaml(msg.target, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GimbalControl & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: control_mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "control_mode: ";
    rosidl_generator_traits::value_to_yaml(msg.control_mode, out);
    out << "\n";
  }

  // member: target
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target:\n";
    to_block_style_yaml(msg.target, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GimbalControl & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::GimbalControl & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::GimbalControl & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::GimbalControl>()
{
  return "as2_msgs::msg::GimbalControl";
}

template<>
inline const char * name<as2_msgs::msg::GimbalControl>()
{
  return "as2_msgs/msg/GimbalControl";
}

template<>
struct has_fixed_size<as2_msgs::msg::GimbalControl>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Vector3Stamped>::value> {};

template<>
struct has_bounded_size<as2_msgs::msg::GimbalControl>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Vector3Stamped>::value> {};

template<>
struct is_message<as2_msgs::msg::GimbalControl>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__TRAITS_HPP_
