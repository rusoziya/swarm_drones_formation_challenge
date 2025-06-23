// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/PlatformStateMachineEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/platform_state_machine_event__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PlatformStateMachineEvent & msg,
  std::ostream & out)
{
  out << "{";
  // member: event
  {
    out << "event: ";
    rosidl_generator_traits::value_to_yaml(msg.event, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PlatformStateMachineEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: event
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "event: ";
    rosidl_generator_traits::value_to_yaml(msg.event, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PlatformStateMachineEvent & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::PlatformStateMachineEvent & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::PlatformStateMachineEvent & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::PlatformStateMachineEvent>()
{
  return "as2_msgs::msg::PlatformStateMachineEvent";
}

template<>
inline const char * name<as2_msgs::msg::PlatformStateMachineEvent>()
{
  return "as2_msgs/msg/PlatformStateMachineEvent";
}

template<>
struct has_fixed_size<as2_msgs::msg::PlatformStateMachineEvent>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::msg::PlatformStateMachineEvent>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::msg::PlatformStateMachineEvent>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__TRAITS_HPP_
