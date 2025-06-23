// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/TrajectoryPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/trajectory_point__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'position'
// Member 'twist'
// Member 'acceleration'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const TrajectoryPoint & msg,
  std::ostream & out)
{
  out << "{";
  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
    out << ", ";
  }

  // member: twist
  {
    out << "twist: ";
    to_flow_style_yaml(msg.twist, out);
    out << ", ";
  }

  // member: acceleration
  {
    out << "acceleration: ";
    to_flow_style_yaml(msg.acceleration, out);
    out << ", ";
  }

  // member: yaw_angle
  {
    out << "yaw_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_angle, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const TrajectoryPoint & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }

  // member: twist
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "twist:\n";
    to_block_style_yaml(msg.twist, out, indentation + 2);
  }

  // member: acceleration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "acceleration:\n";
    to_block_style_yaml(msg.acceleration, out, indentation + 2);
  }

  // member: yaw_angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw_angle: ";
    rosidl_generator_traits::value_to_yaml(msg.yaw_angle, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TrajectoryPoint & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::TrajectoryPoint & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::TrajectoryPoint & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::TrajectoryPoint>()
{
  return "as2_msgs::msg::TrajectoryPoint";
}

template<>
inline const char * name<as2_msgs::msg::TrajectoryPoint>()
{
  return "as2_msgs/msg/TrajectoryPoint";
}

template<>
struct has_fixed_size<as2_msgs::msg::TrajectoryPoint>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct has_bounded_size<as2_msgs::msg::TrajectoryPoint>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct is_message<as2_msgs::msg::TrajectoryPoint>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__TRAITS_HPP_
