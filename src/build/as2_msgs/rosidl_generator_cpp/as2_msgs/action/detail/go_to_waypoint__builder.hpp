// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/GoToWaypoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__GO_TO_WAYPOINT__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__GO_TO_WAYPOINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/go_to_waypoint__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_Goal_max_speed
{
public:
  explicit Init_GoToWaypoint_Goal_max_speed(::as2_msgs::action::GoToWaypoint_Goal & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::GoToWaypoint_Goal max_speed(::as2_msgs::action::GoToWaypoint_Goal::_max_speed_type arg)
  {
    msg_.max_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_Goal msg_;
};

class Init_GoToWaypoint_Goal_target_pose
{
public:
  explicit Init_GoToWaypoint_Goal_target_pose(::as2_msgs::action::GoToWaypoint_Goal & msg)
  : msg_(msg)
  {}
  Init_GoToWaypoint_Goal_max_speed target_pose(::as2_msgs::action::GoToWaypoint_Goal::_target_pose_type arg)
  {
    msg_.target_pose = std::move(arg);
    return Init_GoToWaypoint_Goal_max_speed(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_Goal msg_;
};

class Init_GoToWaypoint_Goal_yaw
{
public:
  Init_GoToWaypoint_Goal_yaw()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GoToWaypoint_Goal_target_pose yaw(::as2_msgs::action::GoToWaypoint_Goal::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_GoToWaypoint_Goal_target_pose(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_Goal>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_Goal_yaw();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_Result_go_to_success
{
public:
  Init_GoToWaypoint_Result_go_to_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::GoToWaypoint_Result go_to_success(::as2_msgs::action::GoToWaypoint_Result::_go_to_success_type arg)
  {
    msg_.go_to_success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_Result>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_Result_go_to_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_Feedback_actual_distance_to_goal
{
public:
  explicit Init_GoToWaypoint_Feedback_actual_distance_to_goal(::as2_msgs::action::GoToWaypoint_Feedback & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::GoToWaypoint_Feedback actual_distance_to_goal(::as2_msgs::action::GoToWaypoint_Feedback::_actual_distance_to_goal_type arg)
  {
    msg_.actual_distance_to_goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_Feedback msg_;
};

class Init_GoToWaypoint_Feedback_actual_speed
{
public:
  Init_GoToWaypoint_Feedback_actual_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GoToWaypoint_Feedback_actual_distance_to_goal actual_speed(::as2_msgs::action::GoToWaypoint_Feedback::_actual_speed_type arg)
  {
    msg_.actual_speed = std::move(arg);
    return Init_GoToWaypoint_Feedback_actual_distance_to_goal(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_Feedback>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_Feedback_actual_speed();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_SendGoal_Request_goal
{
public:
  explicit Init_GoToWaypoint_SendGoal_Request_goal(::as2_msgs::action::GoToWaypoint_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::GoToWaypoint_SendGoal_Request goal(::as2_msgs::action::GoToWaypoint_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_SendGoal_Request msg_;
};

class Init_GoToWaypoint_SendGoal_Request_goal_id
{
public:
  Init_GoToWaypoint_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GoToWaypoint_SendGoal_Request_goal goal_id(::as2_msgs::action::GoToWaypoint_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_GoToWaypoint_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_SendGoal_Response_stamp
{
public:
  explicit Init_GoToWaypoint_SendGoal_Response_stamp(::as2_msgs::action::GoToWaypoint_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::GoToWaypoint_SendGoal_Response stamp(::as2_msgs::action::GoToWaypoint_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_SendGoal_Response msg_;
};

class Init_GoToWaypoint_SendGoal_Response_accepted
{
public:
  Init_GoToWaypoint_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GoToWaypoint_SendGoal_Response_stamp accepted(::as2_msgs::action::GoToWaypoint_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_GoToWaypoint_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_GetResult_Request_goal_id
{
public:
  Init_GoToWaypoint_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::GoToWaypoint_GetResult_Request goal_id(::as2_msgs::action::GoToWaypoint_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_GetResult_Response_result
{
public:
  explicit Init_GoToWaypoint_GetResult_Response_result(::as2_msgs::action::GoToWaypoint_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::GoToWaypoint_GetResult_Response result(::as2_msgs::action::GoToWaypoint_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_GetResult_Response msg_;
};

class Init_GoToWaypoint_GetResult_Response_status
{
public:
  Init_GoToWaypoint_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GoToWaypoint_GetResult_Response_result status(::as2_msgs::action::GoToWaypoint_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_GoToWaypoint_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_GoToWaypoint_FeedbackMessage_feedback
{
public:
  explicit Init_GoToWaypoint_FeedbackMessage_feedback(::as2_msgs::action::GoToWaypoint_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::GoToWaypoint_FeedbackMessage feedback(::as2_msgs::action::GoToWaypoint_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_FeedbackMessage msg_;
};

class Init_GoToWaypoint_FeedbackMessage_goal_id
{
public:
  Init_GoToWaypoint_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GoToWaypoint_FeedbackMessage_feedback goal_id(::as2_msgs::action::GoToWaypoint_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_GoToWaypoint_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::GoToWaypoint_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::GoToWaypoint_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_GoToWaypoint_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__GO_TO_WAYPOINT__BUILDER_HPP_
