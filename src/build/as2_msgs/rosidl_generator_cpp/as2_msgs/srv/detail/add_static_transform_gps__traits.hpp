// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:srv/AddStaticTransformGps.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__TRAITS_HPP_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/srv/detail/add_static_transform_gps__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'gps_position'
#include "sensor_msgs/msg/detail/nav_sat_fix__traits.hpp"

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const AddStaticTransformGps_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: frame_id
  {
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << ", ";
  }

  // member: child_frame_id
  {
    out << "child_frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.child_frame_id, out);
    out << ", ";
  }

  // member: gps_position
  {
    out << "gps_position: ";
    to_flow_style_yaml(msg.gps_position, out);
    out << ", ";
  }

  // member: azimuth
  {
    out << "azimuth: ";
    rosidl_generator_traits::value_to_yaml(msg.azimuth, out);
    out << ", ";
  }

  // member: elevation
  {
    out << "elevation: ";
    rosidl_generator_traits::value_to_yaml(msg.elevation, out);
    out << ", ";
  }

  // member: bank
  {
    out << "bank: ";
    rosidl_generator_traits::value_to_yaml(msg.bank, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const AddStaticTransformGps_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: frame_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << "\n";
  }

  // member: child_frame_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "child_frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.child_frame_id, out);
    out << "\n";
  }

  // member: gps_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gps_position:\n";
    to_block_style_yaml(msg.gps_position, out, indentation + 2);
  }

  // member: azimuth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "azimuth: ";
    rosidl_generator_traits::value_to_yaml(msg.azimuth, out);
    out << "\n";
  }

  // member: elevation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "elevation: ";
    rosidl_generator_traits::value_to_yaml(msg.elevation, out);
    out << "\n";
  }

  // member: bank
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bank: ";
    rosidl_generator_traits::value_to_yaml(msg.bank, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const AddStaticTransformGps_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::srv::AddStaticTransformGps_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::AddStaticTransformGps_Request & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::AddStaticTransformGps_Request>()
{
  return "as2_msgs::srv::AddStaticTransformGps_Request";
}

template<>
inline const char * name<as2_msgs::srv::AddStaticTransformGps_Request>()
{
  return "as2_msgs/srv/AddStaticTransformGps_Request";
}

template<>
struct has_fixed_size<as2_msgs::srv::AddStaticTransformGps_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::srv::AddStaticTransformGps_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::srv::AddStaticTransformGps_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const AddStaticTransformGps_Response & msg,
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
  const AddStaticTransformGps_Response & msg,
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

inline std::string to_yaml(const AddStaticTransformGps_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::srv::AddStaticTransformGps_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::AddStaticTransformGps_Response & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::AddStaticTransformGps_Response>()
{
  return "as2_msgs::srv::AddStaticTransformGps_Response";
}

template<>
inline const char * name<as2_msgs::srv::AddStaticTransformGps_Response>()
{
  return "as2_msgs/srv/AddStaticTransformGps_Response";
}

template<>
struct has_fixed_size<as2_msgs::srv::AddStaticTransformGps_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::srv::AddStaticTransformGps_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::srv::AddStaticTransformGps_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::srv::AddStaticTransformGps>()
{
  return "as2_msgs::srv::AddStaticTransformGps";
}

template<>
inline const char * name<as2_msgs::srv::AddStaticTransformGps>()
{
  return "as2_msgs/srv/AddStaticTransformGps";
}

template<>
struct has_fixed_size<as2_msgs::srv::AddStaticTransformGps>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::srv::AddStaticTransformGps_Request>::value &&
    has_fixed_size<as2_msgs::srv::AddStaticTransformGps_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::srv::AddStaticTransformGps>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::srv::AddStaticTransformGps_Request>::value &&
    has_bounded_size<as2_msgs::srv::AddStaticTransformGps_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::srv::AddStaticTransformGps>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::srv::AddStaticTransformGps_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::srv::AddStaticTransformGps_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__TRAITS_HPP_
