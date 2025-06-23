// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/FollowPath.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__FOLLOW_PATH__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__FOLLOW_PATH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/follow_path__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_Goal_max_speed
{
public:
  explicit Init_FollowPath_Goal_max_speed(::as2_msgs::action::FollowPath_Goal & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowPath_Goal max_speed(::as2_msgs::action::FollowPath_Goal::_max_speed_type arg)
  {
    msg_.max_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Goal msg_;
};

class Init_FollowPath_Goal_path
{
public:
  explicit Init_FollowPath_Goal_path(::as2_msgs::action::FollowPath_Goal & msg)
  : msg_(msg)
  {}
  Init_FollowPath_Goal_max_speed path(::as2_msgs::action::FollowPath_Goal::_path_type arg)
  {
    msg_.path = std::move(arg);
    return Init_FollowPath_Goal_max_speed(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Goal msg_;
};

class Init_FollowPath_Goal_yaw
{
public:
  explicit Init_FollowPath_Goal_yaw(::as2_msgs::action::FollowPath_Goal & msg)
  : msg_(msg)
  {}
  Init_FollowPath_Goal_path yaw(::as2_msgs::action::FollowPath_Goal::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_FollowPath_Goal_path(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Goal msg_;
};

class Init_FollowPath_Goal_header
{
public:
  Init_FollowPath_Goal_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowPath_Goal_yaw header(::as2_msgs::action::FollowPath_Goal::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_FollowPath_Goal_yaw(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_Goal>()
{
  return as2_msgs::action::builder::Init_FollowPath_Goal_header();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_Result_follow_path_success
{
public:
  Init_FollowPath_Result_follow_path_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::FollowPath_Result follow_path_success(::as2_msgs::action::FollowPath_Result::_follow_path_success_type arg)
  {
    msg_.follow_path_success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_Result>()
{
  return as2_msgs::action::builder::Init_FollowPath_Result_follow_path_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_Feedback_next_waypoint_id
{
public:
  explicit Init_FollowPath_Feedback_next_waypoint_id(::as2_msgs::action::FollowPath_Feedback & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowPath_Feedback next_waypoint_id(::as2_msgs::action::FollowPath_Feedback::_next_waypoint_id_type arg)
  {
    msg_.next_waypoint_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Feedback msg_;
};

class Init_FollowPath_Feedback_remaining_waypoints
{
public:
  explicit Init_FollowPath_Feedback_remaining_waypoints(::as2_msgs::action::FollowPath_Feedback & msg)
  : msg_(msg)
  {}
  Init_FollowPath_Feedback_next_waypoint_id remaining_waypoints(::as2_msgs::action::FollowPath_Feedback::_remaining_waypoints_type arg)
  {
    msg_.remaining_waypoints = std::move(arg);
    return Init_FollowPath_Feedback_next_waypoint_id(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Feedback msg_;
};

class Init_FollowPath_Feedback_actual_distance_to_next_waypoint
{
public:
  explicit Init_FollowPath_Feedback_actual_distance_to_next_waypoint(::as2_msgs::action::FollowPath_Feedback & msg)
  : msg_(msg)
  {}
  Init_FollowPath_Feedback_remaining_waypoints actual_distance_to_next_waypoint(::as2_msgs::action::FollowPath_Feedback::_actual_distance_to_next_waypoint_type arg)
  {
    msg_.actual_distance_to_next_waypoint = std::move(arg);
    return Init_FollowPath_Feedback_remaining_waypoints(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Feedback msg_;
};

class Init_FollowPath_Feedback_actual_speed
{
public:
  Init_FollowPath_Feedback_actual_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowPath_Feedback_actual_distance_to_next_waypoint actual_speed(::as2_msgs::action::FollowPath_Feedback::_actual_speed_type arg)
  {
    msg_.actual_speed = std::move(arg);
    return Init_FollowPath_Feedback_actual_distance_to_next_waypoint(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_Feedback>()
{
  return as2_msgs::action::builder::Init_FollowPath_Feedback_actual_speed();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_SendGoal_Request_goal
{
public:
  explicit Init_FollowPath_SendGoal_Request_goal(::as2_msgs::action::FollowPath_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowPath_SendGoal_Request goal(::as2_msgs::action::FollowPath_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_SendGoal_Request msg_;
};

class Init_FollowPath_SendGoal_Request_goal_id
{
public:
  Init_FollowPath_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowPath_SendGoal_Request_goal goal_id(::as2_msgs::action::FollowPath_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_FollowPath_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_FollowPath_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_SendGoal_Response_stamp
{
public:
  explicit Init_FollowPath_SendGoal_Response_stamp(::as2_msgs::action::FollowPath_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowPath_SendGoal_Response stamp(::as2_msgs::action::FollowPath_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_SendGoal_Response msg_;
};

class Init_FollowPath_SendGoal_Response_accepted
{
public:
  Init_FollowPath_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowPath_SendGoal_Response_stamp accepted(::as2_msgs::action::FollowPath_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_FollowPath_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_FollowPath_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_GetResult_Request_goal_id
{
public:
  Init_FollowPath_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::FollowPath_GetResult_Request goal_id(::as2_msgs::action::FollowPath_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_FollowPath_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_GetResult_Response_result
{
public:
  explicit Init_FollowPath_GetResult_Response_result(::as2_msgs::action::FollowPath_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowPath_GetResult_Response result(::as2_msgs::action::FollowPath_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_GetResult_Response msg_;
};

class Init_FollowPath_GetResult_Response_status
{
public:
  Init_FollowPath_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowPath_GetResult_Response_result status(::as2_msgs::action::FollowPath_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_FollowPath_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_FollowPath_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowPath_FeedbackMessage_feedback
{
public:
  explicit Init_FollowPath_FeedbackMessage_feedback(::as2_msgs::action::FollowPath_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowPath_FeedbackMessage feedback(::as2_msgs::action::FollowPath_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_FeedbackMessage msg_;
};

class Init_FollowPath_FeedbackMessage_goal_id
{
public:
  Init_FollowPath_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowPath_FeedbackMessage_feedback goal_id(::as2_msgs::action::FollowPath_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_FollowPath_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::FollowPath_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowPath_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_FollowPath_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__FOLLOW_PATH__BUILDER_HPP_
