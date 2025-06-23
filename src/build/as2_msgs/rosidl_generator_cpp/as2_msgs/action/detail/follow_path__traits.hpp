// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:action/FollowPath.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__FOLLOW_PATH__TRAITS_HPP_
#define AS2_MSGS__ACTION__DETAIL__FOLLOW_PATH__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/action/detail/follow_path__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'yaw'
#include "as2_msgs/msg/detail/yaw_mode__traits.hpp"
// Member 'path'
#include "as2_msgs/msg/detail/pose_with_id__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const FollowPath_Goal & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: yaw
  {
    out << "yaw: ";
    to_flow_style_yaml(msg.yaw, out);
    out << ", ";
  }

  // member: path
  {
    if (msg.path.size() == 0) {
      out << "path: []";
    } else {
      out << "path: [";
      size_t pending_items = msg.path.size();
      for (auto item : msg.path) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: max_speed
  {
    out << "max_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.max_speed, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const FollowPath_Goal & msg,
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

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw:\n";
    to_block_style_yaml(msg.yaw, out, indentation + 2);
  }

  // member: path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.path.size() == 0) {
      out << "path: []\n";
    } else {
      out << "path:\n";
      for (auto item : msg.path) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: max_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.max_speed, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const FollowPath_Goal & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_Goal & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_Goal & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_Goal>()
{
  return "as2_msgs::action::FollowPath_Goal";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_Goal>()
{
  return "as2_msgs/action/FollowPath_Goal";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_Goal>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_Goal>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::action::FollowPath_Goal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const FollowPath_Result & msg,
  std::ostream & out)
{
  out << "{";
  // member: follow_path_success
  {
    out << "follow_path_success: ";
    rosidl_generator_traits::value_to_yaml(msg.follow_path_success, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const FollowPath_Result & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: follow_path_success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "follow_path_success: ";
    rosidl_generator_traits::value_to_yaml(msg.follow_path_success, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const FollowPath_Result & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_Result & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_Result & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_Result>()
{
  return "as2_msgs::action::FollowPath_Result";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_Result>()
{
  return "as2_msgs/action/FollowPath_Result";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_Result>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_Result>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::action::FollowPath_Result>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const FollowPath_Feedback & msg,
  std::ostream & out)
{
  out << "{";
  // member: actual_speed
  {
    out << "actual_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_speed, out);
    out << ", ";
  }

  // member: actual_distance_to_next_waypoint
  {
    out << "actual_distance_to_next_waypoint: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_distance_to_next_waypoint, out);
    out << ", ";
  }

  // member: remaining_waypoints
  {
    out << "remaining_waypoints: ";
    rosidl_generator_traits::value_to_yaml(msg.remaining_waypoints, out);
    out << ", ";
  }

  // member: next_waypoint_id
  {
    out << "next_waypoint_id: ";
    rosidl_generator_traits::value_to_yaml(msg.next_waypoint_id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const FollowPath_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: actual_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "actual_speed: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_speed, out);
    out << "\n";
  }

  // member: actual_distance_to_next_waypoint
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "actual_distance_to_next_waypoint: ";
    rosidl_generator_traits::value_to_yaml(msg.actual_distance_to_next_waypoint, out);
    out << "\n";
  }

  // member: remaining_waypoints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "remaining_waypoints: ";
    rosidl_generator_traits::value_to_yaml(msg.remaining_waypoints, out);
    out << "\n";
  }

  // member: next_waypoint_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "next_waypoint_id: ";
    rosidl_generator_traits::value_to_yaml(msg.next_waypoint_id, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const FollowPath_Feedback & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_Feedback & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_Feedback>()
{
  return "as2_msgs::action::FollowPath_Feedback";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_Feedback>()
{
  return "as2_msgs/action/FollowPath_Feedback";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_Feedback>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_Feedback>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<as2_msgs::action::FollowPath_Feedback>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'goal'
#include "as2_msgs/action/detail/follow_path__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const FollowPath_SendGoal_Request & msg,
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
  const FollowPath_SendGoal_Request & msg,
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

inline std::string to_yaml(const FollowPath_SendGoal_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_SendGoal_Request & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_SendGoal_Request>()
{
  return "as2_msgs::action::FollowPath_SendGoal_Request";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_SendGoal_Request>()
{
  return "as2_msgs/action/FollowPath_SendGoal_Request";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_SendGoal_Request>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::FollowPath_Goal>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_SendGoal_Request>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::FollowPath_Goal>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::FollowPath_SendGoal_Request>
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
  const FollowPath_SendGoal_Response & msg,
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
  const FollowPath_SendGoal_Response & msg,
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

inline std::string to_yaml(const FollowPath_SendGoal_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_SendGoal_Response & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_SendGoal_Response>()
{
  return "as2_msgs::action::FollowPath_SendGoal_Response";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_SendGoal_Response>()
{
  return "as2_msgs/action/FollowPath_SendGoal_Response";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_SendGoal_Response>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_SendGoal_Response>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<as2_msgs::action::FollowPath_SendGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::action::FollowPath_SendGoal>()
{
  return "as2_msgs::action::FollowPath_SendGoal";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_SendGoal>()
{
  return "as2_msgs/action/FollowPath_SendGoal";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_SendGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::action::FollowPath_SendGoal_Request>::value &&
    has_fixed_size<as2_msgs::action::FollowPath_SendGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_SendGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::action::FollowPath_SendGoal_Request>::value &&
    has_bounded_size<as2_msgs::action::FollowPath_SendGoal_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::action::FollowPath_SendGoal>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::action::FollowPath_SendGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::action::FollowPath_SendGoal_Response>
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
  const FollowPath_GetResult_Request & msg,
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
  const FollowPath_GetResult_Request & msg,
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

inline std::string to_yaml(const FollowPath_GetResult_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_GetResult_Request & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_GetResult_Request>()
{
  return "as2_msgs::action::FollowPath_GetResult_Request";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_GetResult_Request>()
{
  return "as2_msgs/action/FollowPath_GetResult_Request";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_GetResult_Request>
  : std::integral_constant<bool, has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_GetResult_Request>
  : std::integral_constant<bool, has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::FollowPath_GetResult_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/follow_path__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const FollowPath_GetResult_Response & msg,
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
  const FollowPath_GetResult_Response & msg,
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

inline std::string to_yaml(const FollowPath_GetResult_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_GetResult_Response & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_GetResult_Response>()
{
  return "as2_msgs::action::FollowPath_GetResult_Response";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_GetResult_Response>()
{
  return "as2_msgs/action/FollowPath_GetResult_Response";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_GetResult_Response>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::FollowPath_Result>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_GetResult_Response>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::FollowPath_Result>::value> {};

template<>
struct is_message<as2_msgs::action::FollowPath_GetResult_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::action::FollowPath_GetResult>()
{
  return "as2_msgs::action::FollowPath_GetResult";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_GetResult>()
{
  return "as2_msgs/action/FollowPath_GetResult";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_GetResult>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::action::FollowPath_GetResult_Request>::value &&
    has_fixed_size<as2_msgs::action::FollowPath_GetResult_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_GetResult>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::action::FollowPath_GetResult_Request>::value &&
    has_bounded_size<as2_msgs::action::FollowPath_GetResult_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::action::FollowPath_GetResult>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::action::FollowPath_GetResult_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::action::FollowPath_GetResult_Response>
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
// #include "as2_msgs/action/detail/follow_path__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const FollowPath_FeedbackMessage & msg,
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
  const FollowPath_FeedbackMessage & msg,
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

inline std::string to_yaml(const FollowPath_FeedbackMessage & msg, bool use_flow_style = false)
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
  const as2_msgs::action::FollowPath_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::FollowPath_FeedbackMessage & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::FollowPath_FeedbackMessage>()
{
  return "as2_msgs::action::FollowPath_FeedbackMessage";
}

template<>
inline const char * name<as2_msgs::action::FollowPath_FeedbackMessage>()
{
  return "as2_msgs/action/FollowPath_FeedbackMessage";
}

template<>
struct has_fixed_size<as2_msgs::action::FollowPath_FeedbackMessage>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::FollowPath_Feedback>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::FollowPath_FeedbackMessage>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::FollowPath_Feedback>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::FollowPath_FeedbackMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits


namespace rosidl_generator_traits
{

template<>
struct is_action<as2_msgs::action::FollowPath>
  : std::true_type
{
};

template<>
struct is_action_goal<as2_msgs::action::FollowPath_Goal>
  : std::true_type
{
};

template<>
struct is_action_result<as2_msgs::action::FollowPath_Result>
  : std::true_type
{
};

template<>
struct is_action_feedback<as2_msgs::action::FollowPath_Feedback>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits


#endif  // AS2_MSGS__ACTION__DETAIL__FOLLOW_PATH__TRAITS_HPP_
