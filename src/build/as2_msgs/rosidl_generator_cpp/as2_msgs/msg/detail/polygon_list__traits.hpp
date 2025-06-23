// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:msg/PolygonList.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POLYGON_LIST__TRAITS_HPP_
#define AS2_MSGS__MSG__DETAIL__POLYGON_LIST__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/msg/detail/polygon_list__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'polygons'
#include "geometry_msgs/msg/detail/polygon_stamped__traits.hpp"

namespace as2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PolygonList & msg,
  std::ostream & out)
{
  out << "{";
  // member: polygons
  {
    if (msg.polygons.size() == 0) {
      out << "polygons: []";
    } else {
      out << "polygons: [";
      size_t pending_items = msg.polygons.size();
      for (auto item : msg.polygons) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PolygonList & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: polygons
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.polygons.size() == 0) {
      out << "polygons: []\n";
    } else {
      out << "polygons:\n";
      for (auto item : msg.polygons) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PolygonList & msg, bool use_flow_style = false)
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
  const as2_msgs::msg::PolygonList & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::msg::PolygonList & msg)
{
  return as2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::msg::PolygonList>()
{
  return "as2_msgs::msg::PolygonList";
}

template<>
inline const char * name<as2_msgs::msg::PolygonList>()
{
  return "as2_msgs/msg/PolygonList";
}

template<>
struct has_fixed_size<as2_msgs::msg::PolygonList>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::msg::PolygonList>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::msg::PolygonList>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__MSG__DETAIL__POLYGON_LIST__TRAITS_HPP_
