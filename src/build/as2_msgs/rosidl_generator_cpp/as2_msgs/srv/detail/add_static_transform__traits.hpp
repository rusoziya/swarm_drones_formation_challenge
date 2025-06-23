// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:srv/AddStaticTransform.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__TRAITS_HPP_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/srv/detail/add_static_transform__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'transform'
#include "geometry_msgs/msg/detail/transform__traits.hpp"

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const AddStaticTransform_Request & msg,
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

  // member: transform
  {
    out << "transform: ";
    to_flow_style_yaml(msg.transform, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const AddStaticTransform_Request & msg,
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

  // member: transform
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "transform:\n";
    to_block_style_yaml(msg.transform, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const AddStaticTransform_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::srv::AddStaticTransform_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::AddStaticTransform_Request & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::AddStaticTransform_Request>()
{
  return "as2_msgs::srv::AddStaticTransform_Request";
}

template<>
inline const char * name<as2_msgs::srv::AddStaticTransform_Request>()
{
  return "as2_msgs/srv/AddStaticTransform_Request";
}

template<>
struct has_fixed_size<as2_msgs::srv::AddStaticTransform_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::srv::AddStaticTransform_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::srv::AddStaticTransform_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const AddStaticTransform_Response & msg,
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
  const AddStaticTransform_Response & msg,
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

inline std::string to_yaml(const AddStaticTransform_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::srv::AddStaticTransform_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::AddStaticTransform_Response & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::AddStaticTransform_Response>()
{
  return "as2_msgs::srv::AddStaticTransform_Response";
}

template<>
inline const char * name<as2_msgs::srv::AddStaticTransform_Response>()
{
  return "as2_msgs/srv/AddStaticTransform_Response";
}

template<>
struct has_fixed_size<as2_msgs::srv::AddStaticTransform_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::srv::AddStaticTransform_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::srv::AddStaticTransform_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::srv::AddStaticTransform>()
{
  return "as2_msgs::srv::AddStaticTransform";
}

template<>
inline const char * name<as2_msgs::srv::AddStaticTransform>()
{
  return "as2_msgs/srv/AddStaticTransform";
}

template<>
struct has_fixed_size<as2_msgs::srv::AddStaticTransform>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::srv::AddStaticTransform_Request>::value &&
    has_fixed_size<as2_msgs::srv::AddStaticTransform_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::srv::AddStaticTransform>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::srv::AddStaticTransform_Request>::value &&
    has_bounded_size<as2_msgs::srv::AddStaticTransform_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::srv::AddStaticTransform>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::srv::AddStaticTransform_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::srv::AddStaticTransform_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__TRAITS_HPP_
