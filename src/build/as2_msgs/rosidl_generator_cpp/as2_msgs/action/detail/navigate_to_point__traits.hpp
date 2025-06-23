// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:action/NavigateToPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__TRAITS_HPP_
#define AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/action/detail/navigate_to_point__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'point'
#include "geometry_msgs/msg/detail/point_stamped__traits.hpp"
// Member 'yaw'
#include "as2_msgs/msg/detail/yaw_mode__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_Goal & msg,
  std::ostream & out)
{
  out << "{";
  // member: point
  {
    out << "point: ";
    to_flow_style_yaml(msg.point, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    to_flow_style_yaml(msg.yaw, out);
    out << ", ";
  }

  // member: navigation_speed
  {
    out << "navigation_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.navigation_speed, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_Goal & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: point
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "point:\n";
    to_block_style_yaml(msg.point, out, indentation + 2);
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw:\n";
    to_block_style_yaml(msg.yaw, out, indentation + 2);
  }

  // member: navigation_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "navigation_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.navigation_speed, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_Goal & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_Goal & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_Goal & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_Goal>()
{
  return "as2_msgs::action::NavigateToPoint_Goal";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_Goal>()
{
  return "as2_msgs/action/NavigateToPoint_Goal";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_Goal>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::msg::YawMode>::value && has_fixed_size<geometry_msgs::msg::PointStamped>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_Goal>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::msg::YawMode>::value && has_bounded_size<geometry_msgs::msg::PointStamped>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_Goal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_Result & msg,
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
  const NavigateToPoint_Result & msg,
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

inline std::string to_yaml(const NavigateToPoint_Result & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_Result & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_Result & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_Result>()
{
  return "as2_msgs::action::NavigateToPoint_Result";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_Result>()
{
  return "as2_msgs/action/NavigateToPoint_Result";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_Result>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_Result>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_Result>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose_stamped__traits.hpp"
// Member 'current_speed'
#include "geometry_msgs/msg/detail/twist_stamped__traits.hpp"
// Member 'navigation_time'
// Member 'estimated_time_remaining'
#include "builtin_interfaces/msg/detail/duration__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_Feedback & msg,
  std::ostream & out)
{
  out << "{";
  // member: current_pose
  {
    out << "current_pose: ";
    to_flow_style_yaml(msg.current_pose, out);
    out << ", ";
  }

  // member: current_speed
  {
    out << "current_speed: ";
    to_flow_style_yaml(msg.current_speed, out);
    out << ", ";
  }

  // member: navigation_time
  {
    out << "navigation_time: ";
    to_flow_style_yaml(msg.navigation_time, out);
    out << ", ";
  }

  // member: estimated_time_remaining
  {
    out << "estimated_time_remaining: ";
    to_flow_style_yaml(msg.estimated_time_remaining, out);
    out << ", ";
  }

  // member: distance_remaining
  {
    out << "distance_remaining: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_remaining, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: current_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_pose:\n";
    to_block_style_yaml(msg.current_pose, out, indentation + 2);
  }

  // member: current_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_speed:\n";
    to_block_style_yaml(msg.current_speed, out, indentation + 2);
  }

  // member: navigation_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "navigation_time:\n";
    to_block_style_yaml(msg.navigation_time, out, indentation + 2);
  }

  // member: estimated_time_remaining
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "estimated_time_remaining:\n";
    to_block_style_yaml(msg.estimated_time_remaining, out, indentation + 2);
  }

  // member: distance_remaining
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "distance_remaining: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_remaining, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_Feedback & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_Feedback & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_Feedback>()
{
  return "as2_msgs::action::NavigateToPoint_Feedback";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_Feedback>()
{
  return "as2_msgs/action/NavigateToPoint_Feedback";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_Feedback>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Duration>::value && has_fixed_size<geometry_msgs::msg::PoseStamped>::value && has_fixed_size<geometry_msgs::msg::TwistStamped>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_Feedback>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Duration>::value && has_bounded_size<geometry_msgs::msg::PoseStamped>::value && has_bounded_size<geometry_msgs::msg::TwistStamped>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_Feedback>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'goal'
#include "as2_msgs/action/detail/navigate_to_point__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_SendGoal_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_id
  {
    out << "goal_id: ";
    to_flow_style_yaml(msg.goal_id, out);
    out << ", ";
  }

  // member: goal
  {
    out << "goal: ";
    to_flow_style_yaml(msg.goal, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_block_style_yaml(msg.goal_id, out, indentation + 2);
  }

  // member: goal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal:\n";
    to_block_style_yaml(msg.goal, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_SendGoal_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_SendGoal_Request & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_SendGoal_Request>()
{
  return "as2_msgs::action::NavigateToPoint_SendGoal_Request";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_SendGoal_Request>()
{
  return "as2_msgs/action/NavigateToPoint_SendGoal_Request";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_SendGoal_Request>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::NavigateToPoint_Goal>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_SendGoal_Request>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::NavigateToPoint_Goal>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_SendGoal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_SendGoal_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: accepted
  {
    out << "accepted: ";
    rosidl_generator_traits::value_to_yaml(msg.accepted, out);
    out << ", ";
  }

  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: accepted
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "accepted: ";
    rosidl_generator_traits::value_to_yaml(msg.accepted, out);
    out << "\n";
  }

  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_SendGoal_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_SendGoal_Response & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_SendGoal_Response>()
{
  return "as2_msgs::action::NavigateToPoint_SendGoal_Response";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_SendGoal_Response>()
{
  return "as2_msgs/action/NavigateToPoint_SendGoal_Response";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_SendGoal_Response>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_SendGoal_Response>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_SendGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_SendGoal>()
{
  return "as2_msgs::action::NavigateToPoint_SendGoal";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_SendGoal>()
{
  return "as2_msgs/action/NavigateToPoint_SendGoal";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_SendGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::action::NavigateToPoint_SendGoal_Request>::value &&
    has_fixed_size<as2_msgs::action::NavigateToPoint_SendGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_SendGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::action::NavigateToPoint_SendGoal_Request>::value &&
    has_bounded_size<as2_msgs::action::NavigateToPoint_SendGoal_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::action::NavigateToPoint_SendGoal>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::action::NavigateToPoint_SendGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::action::NavigateToPoint_SendGoal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_GetResult_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_id
  {
    out << "goal_id: ";
    to_flow_style_yaml(msg.goal_id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_block_style_yaml(msg.goal_id, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_GetResult_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_GetResult_Request & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_GetResult_Request>()
{
  return "as2_msgs::action::NavigateToPoint_GetResult_Request";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_GetResult_Request>()
{
  return "as2_msgs/action/NavigateToPoint_GetResult_Request";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_GetResult_Request>
  : std::integral_constant<bool, has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_GetResult_Request>
  : std::integral_constant<bool, has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_GetResult_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/navigate_to_point__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_GetResult_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: status
  {
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << ", ";
  }

  // member: result
  {
    out << "result: ";
    to_flow_style_yaml(msg.result, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: result
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "result:\n";
    to_block_style_yaml(msg.result, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_GetResult_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_GetResult_Response & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_GetResult_Response>()
{
  return "as2_msgs::action::NavigateToPoint_GetResult_Response";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_GetResult_Response>()
{
  return "as2_msgs/action/NavigateToPoint_GetResult_Response";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_GetResult_Response>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::NavigateToPoint_Result>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_GetResult_Response>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::NavigateToPoint_Result>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_GetResult_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_GetResult>()
{
  return "as2_msgs::action::NavigateToPoint_GetResult";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_GetResult>()
{
  return "as2_msgs/action/NavigateToPoint_GetResult";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_GetResult>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::action::NavigateToPoint_GetResult_Request>::value &&
    has_fixed_size<as2_msgs::action::NavigateToPoint_GetResult_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_GetResult>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::action::NavigateToPoint_GetResult_Request>::value &&
    has_bounded_size<as2_msgs::action::NavigateToPoint_GetResult_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::action::NavigateToPoint_GetResult>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::action::NavigateToPoint_GetResult_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::action::NavigateToPoint_GetResult_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/navigate_to_point__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const NavigateToPoint_FeedbackMessage & msg,
  std::ostream & out)
{
  out << "{";
  // member: goal_id
  {
    out << "goal_id: ";
    to_flow_style_yaml(msg.goal_id, out);
    out << ", ";
  }

  // member: feedback
  {
    out << "feedback: ";
    to_flow_style_yaml(msg.feedback, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigateToPoint_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_block_style_yaml(msg.goal_id, out, indentation + 2);
  }

  // member: feedback
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "feedback:\n";
    to_block_style_yaml(msg.feedback, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigateToPoint_FeedbackMessage & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace action

}  // namespace as2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use as2_msgs::action::to_block_style_yaml() instead")]]
inline void to_yaml(
  const as2_msgs::action::NavigateToPoint_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::NavigateToPoint_FeedbackMessage & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::NavigateToPoint_FeedbackMessage>()
{
  return "as2_msgs::action::NavigateToPoint_FeedbackMessage";
}

template<>
inline const char * name<as2_msgs::action::NavigateToPoint_FeedbackMessage>()
{
  return "as2_msgs/action/NavigateToPoint_FeedbackMessage";
}

template<>
struct has_fixed_size<as2_msgs::action::NavigateToPoint_FeedbackMessage>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::NavigateToPoint_Feedback>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::NavigateToPoint_FeedbackMessage>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::NavigateToPoint_Feedback>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::NavigateToPoint_FeedbackMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits


namespace rosidl_generator_traits
{

template<>
struct is_action<as2_msgs::action::NavigateToPoint>
  : std::true_type
{
};

template<>
struct is_action_goal<as2_msgs::action::NavigateToPoint_Goal>
  : std::true_type
{
};

template<>
struct is_action_result<as2_msgs::action::NavigateToPoint_Result>
  : std::true_type
{
};

template<>
struct is_action_feedback<as2_msgs::action::NavigateToPoint_Feedback>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits


#endif  // AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__TRAITS_HPP_
