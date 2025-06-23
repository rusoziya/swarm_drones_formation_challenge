// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from as2_msgs:action/SetOffboardMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__SET_OFFBOARD_MODE__TRAITS_HPP_
#define AS2_MSGS__ACTION__DETAIL__SET_OFFBOARD_MODE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "as2_msgs/action/detail/set_offboard_mode__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const SetOffboardMode_Goal & msg,
  std::ostream & out)
{
  out << "{";
  // member: request
  {
    out << "request: ";
    rosidl_generator_traits::value_to_yaml(msg.request, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetOffboardMode_Goal & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "request: ";
    rosidl_generator_traits::value_to_yaml(msg.request, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetOffboardMode_Goal & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_Goal & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_Goal & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_Goal>()
{
  return "as2_msgs::action::SetOffboardMode_Goal";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_Goal>()
{
  return "as2_msgs/action/SetOffboardMode_Goal";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_Goal>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_Goal>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_Goal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const SetOffboardMode_Result & msg,
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
  const SetOffboardMode_Result & msg,
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

inline std::string to_yaml(const SetOffboardMode_Result & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_Result & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_Result & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_Result>()
{
  return "as2_msgs::action::SetOffboardMode_Result";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_Result>()
{
  return "as2_msgs/action/SetOffboardMode_Result";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_Result>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_Result>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_Result>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const SetOffboardMode_Feedback & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetOffboardMode_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetOffboardMode_Feedback & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_Feedback & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_Feedback>()
{
  return "as2_msgs::action::SetOffboardMode_Feedback";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_Feedback>()
{
  return "as2_msgs/action/SetOffboardMode_Feedback";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_Feedback>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_Feedback>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_Feedback>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'goal'
#include "as2_msgs/action/detail/set_offboard_mode__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const SetOffboardMode_SendGoal_Request & msg,
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
  const SetOffboardMode_SendGoal_Request & msg,
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

inline std::string to_yaml(const SetOffboardMode_SendGoal_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_SendGoal_Request & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_SendGoal_Request>()
{
  return "as2_msgs::action::SetOffboardMode_SendGoal_Request";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_SendGoal_Request>()
{
  return "as2_msgs/action/SetOffboardMode_SendGoal_Request";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_SendGoal_Request>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::SetOffboardMode_Goal>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_SendGoal_Request>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::SetOffboardMode_Goal>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_SendGoal_Request>
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
  const SetOffboardMode_SendGoal_Response & msg,
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
  const SetOffboardMode_SendGoal_Response & msg,
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

inline std::string to_yaml(const SetOffboardMode_SendGoal_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_SendGoal_Response & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_SendGoal_Response>()
{
  return "as2_msgs::action::SetOffboardMode_SendGoal_Response";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_SendGoal_Response>()
{
  return "as2_msgs/action/SetOffboardMode_SendGoal_Response";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_SendGoal_Response>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_SendGoal_Response>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_SendGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_SendGoal>()
{
  return "as2_msgs::action::SetOffboardMode_SendGoal";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_SendGoal>()
{
  return "as2_msgs/action/SetOffboardMode_SendGoal";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_SendGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::action::SetOffboardMode_SendGoal_Request>::value &&
    has_fixed_size<as2_msgs::action::SetOffboardMode_SendGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_SendGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::action::SetOffboardMode_SendGoal_Request>::value &&
    has_bounded_size<as2_msgs::action::SetOffboardMode_SendGoal_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::action::SetOffboardMode_SendGoal>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::action::SetOffboardMode_SendGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::action::SetOffboardMode_SendGoal_Response>
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
  const SetOffboardMode_GetResult_Request & msg,
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
  const SetOffboardMode_GetResult_Request & msg,
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

inline std::string to_yaml(const SetOffboardMode_GetResult_Request & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_GetResult_Request & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_GetResult_Request>()
{
  return "as2_msgs::action::SetOffboardMode_GetResult_Request";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_GetResult_Request>()
{
  return "as2_msgs/action/SetOffboardMode_GetResult_Request";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_GetResult_Request>
  : std::integral_constant<bool, has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_GetResult_Request>
  : std::integral_constant<bool, has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_GetResult_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/set_offboard_mode__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const SetOffboardMode_GetResult_Response & msg,
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
  const SetOffboardMode_GetResult_Response & msg,
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

inline std::string to_yaml(const SetOffboardMode_GetResult_Response & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_GetResult_Response & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_GetResult_Response>()
{
  return "as2_msgs::action::SetOffboardMode_GetResult_Response";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_GetResult_Response>()
{
  return "as2_msgs/action/SetOffboardMode_GetResult_Response";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_GetResult_Response>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::SetOffboardMode_Result>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_GetResult_Response>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::SetOffboardMode_Result>::value> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_GetResult_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_GetResult>()
{
  return "as2_msgs::action::SetOffboardMode_GetResult";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_GetResult>()
{
  return "as2_msgs/action/SetOffboardMode_GetResult";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_GetResult>
  : std::integral_constant<
    bool,
    has_fixed_size<as2_msgs::action::SetOffboardMode_GetResult_Request>::value &&
    has_fixed_size<as2_msgs::action::SetOffboardMode_GetResult_Response>::value
  >
{
};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_GetResult>
  : std::integral_constant<
    bool,
    has_bounded_size<as2_msgs::action::SetOffboardMode_GetResult_Request>::value &&
    has_bounded_size<as2_msgs::action::SetOffboardMode_GetResult_Response>::value
  >
{
};

template<>
struct is_service<as2_msgs::action::SetOffboardMode_GetResult>
  : std::true_type
{
};

template<>
struct is_service_request<as2_msgs::action::SetOffboardMode_GetResult_Request>
  : std::true_type
{
};

template<>
struct is_service_response<as2_msgs::action::SetOffboardMode_GetResult_Response>
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
// #include "as2_msgs/action/detail/set_offboard_mode__traits.hpp"

namespace as2_msgs
{

namespace action
{

inline void to_flow_style_yaml(
  const SetOffboardMode_FeedbackMessage & msg,
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
  const SetOffboardMode_FeedbackMessage & msg,
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

inline std::string to_yaml(const SetOffboardMode_FeedbackMessage & msg, bool use_flow_style = false)
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
  const as2_msgs::action::SetOffboardMode_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  as2_msgs::action::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use as2_msgs::action::to_yaml() instead")]]
inline std::string to_yaml(const as2_msgs::action::SetOffboardMode_FeedbackMessage & msg)
{
  return as2_msgs::action::to_yaml(msg);
}

template<>
inline const char * data_type<as2_msgs::action::SetOffboardMode_FeedbackMessage>()
{
  return "as2_msgs::action::SetOffboardMode_FeedbackMessage";
}

template<>
inline const char * name<as2_msgs::action::SetOffboardMode_FeedbackMessage>()
{
  return "as2_msgs/action/SetOffboardMode_FeedbackMessage";
}

template<>
struct has_fixed_size<as2_msgs::action::SetOffboardMode_FeedbackMessage>
  : std::integral_constant<bool, has_fixed_size<as2_msgs::action::SetOffboardMode_Feedback>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<as2_msgs::action::SetOffboardMode_FeedbackMessage>
  : std::integral_constant<bool, has_bounded_size<as2_msgs::action::SetOffboardMode_Feedback>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<as2_msgs::action::SetOffboardMode_FeedbackMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits


namespace rosidl_generator_traits
{

template<>
struct is_action<as2_msgs::action::SetOffboardMode>
  : std::true_type
{
};

template<>
struct is_action_goal<as2_msgs::action::SetOffboardMode_Goal>
  : std::true_type
{
};

template<>
struct is_action_result<as2_msgs::action::SetOffboardMode_Result>
  : std::true_type
{
};

template<>
struct is_action_feedback<as2_msgs::action::SetOffboardMode_Feedback>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits


#endif  // AS2_MSGS__ACTION__DETAIL__SET_OFFBOARD_MODE__TRAITS_HPP_
