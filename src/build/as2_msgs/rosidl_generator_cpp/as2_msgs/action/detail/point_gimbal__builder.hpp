// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/PointGimbal.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/point_gimbal__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_Goal_follow_mode
{
public:
  explicit Init_PointGimbal_Goal_follow_mode(::as2_msgs::action::PointGimbal_Goal & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::PointGimbal_Goal follow_mode(::as2_msgs::action::PointGimbal_Goal::_follow_mode_type arg)
  {
    msg_.follow_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_Goal msg_;
};

class Init_PointGimbal_Goal_control
{
public:
  Init_PointGimbal_Goal_control()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointGimbal_Goal_follow_mode control(::as2_msgs::action::PointGimbal_Goal::_control_type arg)
  {
    msg_.control = std::move(arg);
    return Init_PointGimbal_Goal_follow_mode(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_Goal>()
{
  return as2_msgs::action::builder::Init_PointGimbal_Goal_control();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_Result_success
{
public:
  Init_PointGimbal_Result_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::PointGimbal_Result success(::as2_msgs::action::PointGimbal_Result::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_Result>()
{
  return as2_msgs::action::builder::Init_PointGimbal_Result_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_Feedback_gimbal_speed
{
public:
  explicit Init_PointGimbal_Feedback_gimbal_speed(::as2_msgs::action::PointGimbal_Feedback & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::PointGimbal_Feedback gimbal_speed(::as2_msgs::action::PointGimbal_Feedback::_gimbal_speed_type arg)
  {
    msg_.gimbal_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_Feedback msg_;
};

class Init_PointGimbal_Feedback_gimbal_attitude
{
public:
  Init_PointGimbal_Feedback_gimbal_attitude()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointGimbal_Feedback_gimbal_speed gimbal_attitude(::as2_msgs::action::PointGimbal_Feedback::_gimbal_attitude_type arg)
  {
    msg_.gimbal_attitude = std::move(arg);
    return Init_PointGimbal_Feedback_gimbal_speed(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_Feedback>()
{
  return as2_msgs::action::builder::Init_PointGimbal_Feedback_gimbal_attitude();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_SendGoal_Request_goal
{
public:
  explicit Init_PointGimbal_SendGoal_Request_goal(::as2_msgs::action::PointGimbal_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::PointGimbal_SendGoal_Request goal(::as2_msgs::action::PointGimbal_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_SendGoal_Request msg_;
};

class Init_PointGimbal_SendGoal_Request_goal_id
{
public:
  Init_PointGimbal_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointGimbal_SendGoal_Request_goal goal_id(::as2_msgs::action::PointGimbal_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_PointGimbal_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_PointGimbal_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_SendGoal_Response_stamp
{
public:
  explicit Init_PointGimbal_SendGoal_Response_stamp(::as2_msgs::action::PointGimbal_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::PointGimbal_SendGoal_Response stamp(::as2_msgs::action::PointGimbal_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_SendGoal_Response msg_;
};

class Init_PointGimbal_SendGoal_Response_accepted
{
public:
  Init_PointGimbal_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointGimbal_SendGoal_Response_stamp accepted(::as2_msgs::action::PointGimbal_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_PointGimbal_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_PointGimbal_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_GetResult_Request_goal_id
{
public:
  Init_PointGimbal_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::PointGimbal_GetResult_Request goal_id(::as2_msgs::action::PointGimbal_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_PointGimbal_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_GetResult_Response_result
{
public:
  explicit Init_PointGimbal_GetResult_Response_result(::as2_msgs::action::PointGimbal_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::PointGimbal_GetResult_Response result(::as2_msgs::action::PointGimbal_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_GetResult_Response msg_;
};

class Init_PointGimbal_GetResult_Response_status
{
public:
  Init_PointGimbal_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointGimbal_GetResult_Response_result status(::as2_msgs::action::PointGimbal_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_PointGimbal_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_PointGimbal_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_PointGimbal_FeedbackMessage_feedback
{
public:
  explicit Init_PointGimbal_FeedbackMessage_feedback(::as2_msgs::action::PointGimbal_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::PointGimbal_FeedbackMessage feedback(::as2_msgs::action::PointGimbal_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_FeedbackMessage msg_;
};

class Init_PointGimbal_FeedbackMessage_goal_id
{
public:
  Init_PointGimbal_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointGimbal_FeedbackMessage_feedback goal_id(::as2_msgs::action::PointGimbal_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_PointGimbal_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::PointGimbal_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::PointGimbal_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_PointGimbal_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__BUILDER_HPP_
