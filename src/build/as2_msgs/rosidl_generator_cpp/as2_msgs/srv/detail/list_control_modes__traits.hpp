// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:srv/ListControlModes.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__TRAITS_HPP_
#define AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/srv/detail/list_control_modes__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const ListControlModes_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ListControlModes_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ListControlModes_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::srv::ListControlModes_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::ListControlModes_Request & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::ListControlModes_Request>()
{
  return "as2_msgs::srv::ListControlModes_Request";
}

template<>
inline const char * name<as2_msgs::srv::ListControlModes_Request>()
{
  return "as2_msgs/srv/ListControlModes_Request";
}

template<>
struct has_fixed_size<as2_msgs::srv::ListControlModes_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::srv::ListControlModes_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::srv::ListControlModes_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const ListControlModes_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: source
  {
    out << "source: ";
    rosidl_generator_traits::value_to_yaml(msg.source, out);
    out << ", ";
  }

  // member: control_modes
  {
    if (msg.control_modes.size() == 0) {
      out << "control_modes: []";
    } else {
      out << "control_modes: [";
      size_t pending_items = msg.control_modes.size();
      for (auto item : msg.control_modes) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const ListControlModes_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: source
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "source: ";
    rosidl_generator_traits::value_to_yaml(msg.source, out);
    out << "\n";
  }

  // member: control_modes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.control_modes.size() == 0) {
      out << "control_modes: []\n";
    } else {
      out << "control_modes:\n";
      for (auto item : msg.control_modes) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ListControlModes_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::srv::ListControlModes_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::srv::ListControlModes_Response & msg)
{
  return as2_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::srv::ListControlModes_Response>()
{
  return "as2_msgs::srv::ListControlModes_Response";
}

template<>
inline const char * name<as2_msgs::srv::ListControlModes_Response>()
{
  return "as2_msgs/srv/ListControlModes_Response";
}

template<>
struct has_fixed_size<as2_msgs::srv::ListControlModes_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::srv::ListControlModes_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::srv::ListControlModes_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::srv::ListControlModes>()
{
  return "as2_msgs::srv::ListControlModes";
}

template<>
inline const char * name<as2_msgs::srv::ListControlModes>()
{
  return "as2_msgs/srv/ListControlModes";
}

template<>
struct has_fixed_size<as2_msgs::srv::ListControlModes>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::srv::ListControlModes_Request>::value &&
    has_fixed_size<as2_msgs::srv::ListControlModes_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::srv::ListControlModes>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::srv::ListControlModes_Request>::value &&
    has_bounded_size<as2_msgs::srv::ListControlModes_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::srv::ListControlModes>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::srv::ListControlModes_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::srv::ListControlModes_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__TRAITS_HPP_
