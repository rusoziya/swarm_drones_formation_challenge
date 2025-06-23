// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/NavigateToPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/navigate_to_point__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_Goal_navigation_speed
{
public:
  explicit Init_NavigateToPoint_Goal_navigation_speed(::as2_msgs::action::NavigateToPoint_Goal & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::NavigateToPoint_Goal navigation_speed(::as2_msgs::action::NavigateToPoint_Goal::_navigation_speed_type arg)
  {
    msg_.navigation_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Goal msg_;
};

class Init_NavigateToPoint_Goal_yaw
{
public:
  explicit Init_NavigateToPoint_Goal_yaw(::as2_msgs::action::NavigateToPoint_Goal & msg)
  : msg_(msg)
  {}
  Init_NavigateToPoint_Goal_navigation_speed yaw(::as2_msgs::action::NavigateToPoint_Goal::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_NavigateToPoint_Goal_navigation_speed(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Goal msg_;
};

class Init_NavigateToPoint_Goal_point
{
public:
  Init_NavigateToPoint_Goal_point()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateToPoint_Goal_yaw point(::as2_msgs::action::NavigateToPoint_Goal::_point_type arg)
  {
    msg_.point = std::move(arg);
    return Init_NavigateToPoint_Goal_yaw(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_Goal>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_Goal_point();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_Result_success
{
public:
  Init_NavigateToPoint_Result_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::NavigateToPoint_Result success(::as2_msgs::action::NavigateToPoint_Result::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_Result>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_Result_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_Feedback_distance_remaining
{
public:
  explicit Init_NavigateToPoint_Feedback_distance_remaining(::as2_msgs::action::NavigateToPoint_Feedback & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::NavigateToPoint_Feedback distance_remaining(::as2_msgs::action::NavigateToPoint_Feedback::_distance_remaining_type arg)
  {
    msg_.distance_remaining = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Feedback msg_;
};

class Init_NavigateToPoint_Feedback_estimated_time_remaining
{
public:
  explicit Init_NavigateToPoint_Feedback_estimated_time_remaining(::as2_msgs::action::NavigateToPoint_Feedback & msg)
  : msg_(msg)
  {}
  Init_NavigateToPoint_Feedback_distance_remaining estimated_time_remaining(::as2_msgs::action::NavigateToPoint_Feedback::_estimated_time_remaining_type arg)
  {
    msg_.estimated_time_remaining = std::move(arg);
    return Init_NavigateToPoint_Feedback_distance_remaining(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Feedback msg_;
};

class Init_NavigateToPoint_Feedback_navigation_time
{
public:
  explicit Init_NavigateToPoint_Feedback_navigation_time(::as2_msgs::action::NavigateToPoint_Feedback & msg)
  : msg_(msg)
  {}
  Init_NavigateToPoint_Feedback_estimated_time_remaining navigation_time(::as2_msgs::action::NavigateToPoint_Feedback::_navigation_time_type arg)
  {
    msg_.navigation_time = std::move(arg);
    return Init_NavigateToPoint_Feedback_estimated_time_remaining(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Feedback msg_;
};

class Init_NavigateToPoint_Feedback_current_speed
{
public:
  explicit Init_NavigateToPoint_Feedback_current_speed(::as2_msgs::action::NavigateToPoint_Feedback & msg)
  : msg_(msg)
  {}
  Init_NavigateToPoint_Feedback_navigation_time current_speed(::as2_msgs::action::NavigateToPoint_Feedback::_current_speed_type arg)
  {
    msg_.current_speed = std::move(arg);
    return Init_NavigateToPoint_Feedback_navigation_time(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Feedback msg_;
};

class Init_NavigateToPoint_Feedback_current_pose
{
public:
  Init_NavigateToPoint_Feedback_current_pose()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateToPoint_Feedback_current_speed current_pose(::as2_msgs::action::NavigateToPoint_Feedback::_current_pose_type arg)
  {
    msg_.current_pose = std::move(arg);
    return Init_NavigateToPoint_Feedback_current_speed(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_Feedback>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_Feedback_current_pose();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_SendGoal_Request_goal
{
public:
  explicit Init_NavigateToPoint_SendGoal_Request_goal(::as2_msgs::action::NavigateToPoint_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::NavigateToPoint_SendGoal_Request goal(::as2_msgs::action::NavigateToPoint_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_SendGoal_Request msg_;
};

class Init_NavigateToPoint_SendGoal_Request_goal_id
{
public:
  Init_NavigateToPoint_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateToPoint_SendGoal_Request_goal goal_id(::as2_msgs::action::NavigateToPoint_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_NavigateToPoint_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_SendGoal_Response_stamp
{
public:
  explicit Init_NavigateToPoint_SendGoal_Response_stamp(::as2_msgs::action::NavigateToPoint_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::NavigateToPoint_SendGoal_Response stamp(::as2_msgs::action::NavigateToPoint_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_SendGoal_Response msg_;
};

class Init_NavigateToPoint_SendGoal_Response_accepted
{
public:
  Init_NavigateToPoint_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateToPoint_SendGoal_Response_stamp accepted(::as2_msgs::action::NavigateToPoint_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_NavigateToPoint_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_GetResult_Request_goal_id
{
public:
  Init_NavigateToPoint_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::NavigateToPoint_GetResult_Request goal_id(::as2_msgs::action::NavigateToPoint_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_GetResult_Response_result
{
public:
  explicit Init_NavigateToPoint_GetResult_Response_result(::as2_msgs::action::NavigateToPoint_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::NavigateToPoint_GetResult_Response result(::as2_msgs::action::NavigateToPoint_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_GetResult_Response msg_;
};

class Init_NavigateToPoint_GetResult_Response_status
{
public:
  Init_NavigateToPoint_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateToPoint_GetResult_Response_result status(::as2_msgs::action::NavigateToPoint_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_NavigateToPoint_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_NavigateToPoint_FeedbackMessage_feedback
{
public:
  explicit Init_NavigateToPoint_FeedbackMessage_feedback(::as2_msgs::action::NavigateToPoint_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::NavigateToPoint_FeedbackMessage feedback(::as2_msgs::action::NavigateToPoint_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_FeedbackMessage msg_;
};

class Init_NavigateToPoint_FeedbackMessage_goal_id
{
public:
  Init_NavigateToPoint_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigateToPoint_FeedbackMessage_feedback goal_id(::as2_msgs::action::NavigateToPoint_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_NavigateToPoint_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::NavigateToPoint_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::NavigateToPoint_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_NavigateToPoint_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__BUILDER_HPP_
