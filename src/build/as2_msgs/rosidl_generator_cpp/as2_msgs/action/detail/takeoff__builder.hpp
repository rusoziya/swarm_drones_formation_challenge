// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/Takeoff.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__TAKEOFF__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__TAKEOFF__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/takeoff__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_Goal_takeoff_speed
{
public:
  explicit Init_Takeoff_Goal_takeoff_speed(::as2_msgs::action::Takeoff_Goal & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::Takeoff_Goal takeoff_speed(::as2_msgs::action::Takeoff_Goal::_takeoff_speed_type arg)
  {
    msg_.takeoff_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_Goal msg_;
};

class Init_Takeoff_Goal_takeoff_height
{
public:
  Init_Takeoff_Goal_takeoff_height()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Takeoff_Goal_takeoff_speed takeoff_height(::as2_msgs::action::Takeoff_Goal::_takeoff_height_type arg)
  {
    msg_.takeoff_height = std::move(arg);
    return Init_Takeoff_Goal_takeoff_speed(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_Goal>()
{
  return as2_msgs::action::builder::Init_Takeoff_Goal_takeoff_height();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_Result_takeoff_success
{
public:
  Init_Takeoff_Result_takeoff_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::Takeoff_Result takeoff_success(::as2_msgs::action::Takeoff_Result::_takeoff_success_type arg)
  {
    msg_.takeoff_success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_Result>()
{
  return as2_msgs::action::builder::Init_Takeoff_Result_takeoff_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_Feedback_actual_takeoff_height
{
public:
  explicit Init_Takeoff_Feedback_actual_takeoff_height(::as2_msgs::action::Takeoff_Feedback & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::Takeoff_Feedback actual_takeoff_height(::as2_msgs::action::Takeoff_Feedback::_actual_takeoff_height_type arg)
  {
    msg_.actual_takeoff_height = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_Feedback msg_;
};

class Init_Takeoff_Feedback_actual_takeoff_speed
{
public:
  Init_Takeoff_Feedback_actual_takeoff_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Takeoff_Feedback_actual_takeoff_height actual_takeoff_speed(::as2_msgs::action::Takeoff_Feedback::_actual_takeoff_speed_type arg)
  {
    msg_.actual_takeoff_speed = std::move(arg);
    return Init_Takeoff_Feedback_actual_takeoff_height(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_Feedback>()
{
  return as2_msgs::action::builder::Init_Takeoff_Feedback_actual_takeoff_speed();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_SendGoal_Request_goal
{
public:
  explicit Init_Takeoff_SendGoal_Request_goal(::as2_msgs::action::Takeoff_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::Takeoff_SendGoal_Request goal(::as2_msgs::action::Takeoff_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_SendGoal_Request msg_;
};

class Init_Takeoff_SendGoal_Request_goal_id
{
public:
  Init_Takeoff_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Takeoff_SendGoal_Request_goal goal_id(::as2_msgs::action::Takeoff_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_Takeoff_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_Takeoff_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_SendGoal_Response_stamp
{
public:
  explicit Init_Takeoff_SendGoal_Response_stamp(::as2_msgs::action::Takeoff_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::Takeoff_SendGoal_Response stamp(::as2_msgs::action::Takeoff_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_SendGoal_Response msg_;
};

class Init_Takeoff_SendGoal_Response_accepted
{
public:
  Init_Takeoff_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Takeoff_SendGoal_Response_stamp accepted(::as2_msgs::action::Takeoff_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_Takeoff_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_Takeoff_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_GetResult_Request_goal_id
{
public:
  Init_Takeoff_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::Takeoff_GetResult_Request goal_id(::as2_msgs::action::Takeoff_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_Takeoff_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_GetResult_Response_result
{
public:
  explicit Init_Takeoff_GetResult_Response_result(::as2_msgs::action::Takeoff_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::Takeoff_GetResult_Response result(::as2_msgs::action::Takeoff_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_GetResult_Response msg_;
};

class Init_Takeoff_GetResult_Response_status
{
public:
  Init_Takeoff_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Takeoff_GetResult_Response_result status(::as2_msgs::action::Takeoff_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_Takeoff_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_Takeoff_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_Takeoff_FeedbackMessage_feedback
{
public:
  explicit Init_Takeoff_FeedbackMessage_feedback(::as2_msgs::action::Takeoff_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::Takeoff_FeedbackMessage feedback(::as2_msgs::action::Takeoff_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_FeedbackMessage msg_;
};

class Init_Takeoff_FeedbackMessage_goal_id
{
public:
  Init_Takeoff_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Takeoff_FeedbackMessage_feedback goal_id(::as2_msgs::action::Takeoff_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_Takeoff_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::Takeoff_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::Takeoff_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_Takeoff_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__TAKEOFF__BUILDER_HPP_
