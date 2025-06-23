// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:srv/SetOrigin.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_ORIGIN__TRAITS_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_ORIGIN__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/srv/detail/set_origin__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'origin'
#include "geographic_msgs/msg/detail/geo_point__traits.hpp"

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetOrigin_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: origin
  {
    out << "origin: ";
    to_flow_style_yaml(msg.origin, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetOrigin_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: origin
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "origin:\n";
    to_block_style_yaml(msg.origin, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetOrigin_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::srv::SetOrigin_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::SetOrigin_Request & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::SetOrigin_Request>()
{
  return "as2_msgs::srv::SetOrigin_Request";
}

template<>
inline const char * name<as2_msgs::srv::SetOrigin_Request>()
{
  return "as2_msgs/srv/SetOrigin_Request";
}

template<>
struct has_fixed_size<as2_msgs::srv::SetOrigin_Request>
  : std::integral_constant<bool, has_fixed_size<geographic_msgs::msg::GeoPoint>::value> {};

template<>
struct has_bounded_size<as2_msgs::srv::SetOrigin_Request>
  : std::integral_constant<bool, has_bounded_size<geographic_msgs::msg::GeoPoint>::value> {};

template<>
struct is_message<as2_msgs::srv::SetOrigin_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetOrigin_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetOrigin_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetOrigin_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::srv::SetOrigin_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::SetOrigin_Response & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::SetOrigin_Response>()
{
  return "as2_msgs::srv::SetOrigin_Response";
}

template<>
inline const char * name<as2_msgs::srv::SetOrigin_Response>()
{
  return "as2_msgs/srv/SetOrigin_Response";
}

template<>
struct has_fixed_size<as2_msgs::srv::SetOrigin_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::srv::SetOrigin_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::srv::SetOrigin_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::srv::SetOrigin>()
{
  return "as2_msgs::srv::SetOrigin";
}

template<>
inline const char * name<as2_msgs::srv::SetOrigin>()
{
  return "as2_msgs/srv/SetOrigin";
}

template<>
struct has_fixed_size<as2_msgs::srv::SetOrigin>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::srv::SetOrigin_Request>::value &&
    has_fixed_size<as2_msgs::srv::SetOrigin_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::srv::SetOrigin>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::srv::SetOrigin_Request>::value &&
    has_bounded_size<as2_msgs::srv::SetOrigin_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::srv::SetOrigin>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::srv::SetOrigin_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::srv::SetOrigin_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__SRV__DETAIL__SET_ORIGIN__TRAITS_HPP_
