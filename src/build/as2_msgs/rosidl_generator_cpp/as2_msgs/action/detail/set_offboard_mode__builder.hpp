// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/SetOffboardMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__SET_OFFBOARD_MODE__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__SET_OFFBOARD_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/set_offboard_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_Goal_request
{
public:
  Init_SetOffboardMode_Goal_request()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::SetOffboardMode_Goal request(::as2_msgs::action::SetOffboardMode_Goal::_request_type arg)
  {
    msg_.request = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_Goal>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_Goal_request();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_Result_success
{
public:
  Init_SetOffboardMode_Result_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::SetOffboardMode_Result success(::as2_msgs::action::SetOffboardMode_Result::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_Result>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_Result_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{


}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_Feedback>()
{
  return ::as2_msgs::action::SetOffboardMode_Feedback(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_SendGoal_Request_goal
{
public:
  explicit Init_SetOffboardMode_SendGoal_Request_goal(::as2_msgs::action::SetOffboardMode_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::SetOffboardMode_SendGoal_Request goal(::as2_msgs::action::SetOffboardMode_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_SendGoal_Request msg_;
};

class Init_SetOffboardMode_SendGoal_Request_goal_id
{
public:
  Init_SetOffboardMode_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetOffboardMode_SendGoal_Request_goal goal_id(::as2_msgs::action::SetOffboardMode_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_SetOffboardMode_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_SendGoal_Response_stamp
{
public:
  explicit Init_SetOffboardMode_SendGoal_Response_stamp(::as2_msgs::action::SetOffboardMode_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::SetOffboardMode_SendGoal_Response stamp(::as2_msgs::action::SetOffboardMode_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_SendGoal_Response msg_;
};

class Init_SetOffboardMode_SendGoal_Response_accepted
{
public:
  Init_SetOffboardMode_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetOffboardMode_SendGoal_Response_stamp accepted(::as2_msgs::action::SetOffboardMode_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_SetOffboardMode_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_GetResult_Request_goal_id
{
public:
  Init_SetOffboardMode_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::SetOffboardMode_GetResult_Request goal_id(::as2_msgs::action::SetOffboardMode_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_GetResult_Response_result
{
public:
  explicit Init_SetOffboardMode_GetResult_Response_result(::as2_msgs::action::SetOffboardMode_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::SetOffboardMode_GetResult_Response result(::as2_msgs::action::SetOffboardMode_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_GetResult_Response msg_;
};

class Init_SetOffboardMode_GetResult_Response_status
{
public:
  Init_SetOffboardMode_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetOffboardMode_GetResult_Response_result status(::as2_msgs::action::SetOffboardMode_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_SetOffboardMode_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_SetOffboardMode_FeedbackMessage_feedback
{
public:
  explicit Init_SetOffboardMode_FeedbackMessage_feedback(::as2_msgs::action::SetOffboardMode_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::SetOffboardMode_FeedbackMessage feedback(::as2_msgs::action::SetOffboardMode_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_FeedbackMessage msg_;
};

class Init_SetOffboardMode_FeedbackMessage_goal_id
{
public:
  Init_SetOffboardMode_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetOffboardMode_FeedbackMessage_feedback goal_id(::as2_msgs::action::SetOffboardMode_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_SetOffboardMode_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::SetOffboardMode_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::SetOffboardMode_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_SetOffboardMode_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__SET_OFFBOARD_MODE__BUILDER_HPP_
