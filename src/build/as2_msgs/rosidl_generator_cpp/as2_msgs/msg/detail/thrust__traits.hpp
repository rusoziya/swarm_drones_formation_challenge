// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/Thrust.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__THRUST__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__THRUST__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/thrust__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Thrust & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: thrust
  {
    out << "thrust: ";
    rosidl_generator_traits::value_to_yaml(msg.thrust, out);
    out << ", ";
  }

  // member: thrust_normalized
  {
    out << "thrust_normalized: ";
    rosidl_generator_traits::value_to_yaml(msg.thrust_normalized, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Thrust & msg,
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

  // member: thrust
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "thrust: ";
    rosidl_generator_traits::value_to_yaml(msg.thrust, out);
    out << "\n";
  }

  // member: thrust_normalized
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "thrust_normalized: ";
    rosidl_generator_traits::value_to_yaml(msg.thrust_normalized, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Thrust & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::Thrust & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::Thrust & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::Thrust>()
{
  return "as2_msgs::msg::Thrust";
}

template<>
inline const char * name<as2_msgs::msg::Thrust>()
{
  return "as2_msgs/msg/Thrust";
}

template<>
struct has_fixed_size<as2_msgs::msg::Thrust>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<as2_msgs::msg::Thrust>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<as2_msgs::msg::Thrust>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__THRUST__TRAITS_HPP_
