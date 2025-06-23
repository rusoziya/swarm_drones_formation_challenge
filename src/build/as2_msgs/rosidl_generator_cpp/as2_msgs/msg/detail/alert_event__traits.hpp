// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/AlertEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ALERT_EVENT__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__ALERT_EVENT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/alert_event__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const AlertEvent & msg,
  std::ostream & out)
{
  out << "{";
  // member: alert
  {
    out << "alert: ";
    rosidl_generator_traits::value_to_yaml(msg.alert, out);
    out << ", ";
  }

  // member: description
  {
    out << "description: ";
    rosidl_generator_traits::value_to_yaml(msg.description, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const AlertEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: alert
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "alert: ";
    rosidl_generator_traits::value_to_yaml(msg.alert, out);
    out << "\n";
  }

  // member: description
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "description: ";
    rosidl_generator_traits::value_to_yaml(msg.description, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const AlertEvent & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::AlertEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::AlertEvent & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::AlertEvent>()
{
  return "as2_msgs::msg::AlertEvent";
}

template<>
inline const char * name<as2_msgs::msg::AlertEvent>()
{
  return "as2_msgs/msg/AlertEvent";
}

template<>
struct has_fixed_size<as2_msgs::msg::AlertEvent>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::msg::AlertEvent>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::msg::AlertEvent>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__ALERT_EVENT__TRAITS_HPP_
