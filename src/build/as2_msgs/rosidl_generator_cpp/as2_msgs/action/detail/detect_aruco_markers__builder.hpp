// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/DetectArucoMarkers.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__DETECT_ARUCO_MARKERS__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__DETECT_ARUCO_MARKERS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/detect_aruco_markers__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_Goal_target_ids
{
public:
  Init_DetectArucoMarkers_Goal_target_ids()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::DetectArucoMarkers_Goal target_ids(::as2_msgs::action::DetectArucoMarkers_Goal::_target_ids_type arg)
  {
    msg_.target_ids = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_Goal>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_Goal_target_ids();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_Result_success
{
public:
  Init_DetectArucoMarkers_Result_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::DetectArucoMarkers_Result success(::as2_msgs::action::DetectArucoMarkers_Result::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_Result>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_Result_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_Feedback_sucess
{
public:
  Init_DetectArucoMarkers_Feedback_sucess()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::DetectArucoMarkers_Feedback sucess(::as2_msgs::action::DetectArucoMarkers_Feedback::_sucess_type arg)
  {
    msg_.sucess = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_Feedback>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_Feedback_sucess();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_SendGoal_Request_goal
{
public:
  explicit Init_DetectArucoMarkers_SendGoal_Request_goal(::as2_msgs::action::DetectArucoMarkers_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::DetectArucoMarkers_SendGoal_Request goal(::as2_msgs::action::DetectArucoMarkers_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_SendGoal_Request msg_;
};

class Init_DetectArucoMarkers_SendGoal_Request_goal_id
{
public:
  Init_DetectArucoMarkers_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectArucoMarkers_SendGoal_Request_goal goal_id(::as2_msgs::action::DetectArucoMarkers_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_DetectArucoMarkers_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_SendGoal_Response_stamp
{
public:
  explicit Init_DetectArucoMarkers_SendGoal_Response_stamp(::as2_msgs::action::DetectArucoMarkers_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::DetectArucoMarkers_SendGoal_Response stamp(::as2_msgs::action::DetectArucoMarkers_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_SendGoal_Response msg_;
};

class Init_DetectArucoMarkers_SendGoal_Response_accepted
{
public:
  Init_DetectArucoMarkers_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectArucoMarkers_SendGoal_Response_stamp accepted(::as2_msgs::action::DetectArucoMarkers_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_DetectArucoMarkers_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_GetResult_Request_goal_id
{
public:
  Init_DetectArucoMarkers_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::DetectArucoMarkers_GetResult_Request goal_id(::as2_msgs::action::DetectArucoMarkers_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_GetResult_Response_result
{
public:
  explicit Init_DetectArucoMarkers_GetResult_Response_result(::as2_msgs::action::DetectArucoMarkers_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::DetectArucoMarkers_GetResult_Response result(::as2_msgs::action::DetectArucoMarkers_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_GetResult_Response msg_;
};

class Init_DetectArucoMarkers_GetResult_Response_status
{
public:
  Init_DetectArucoMarkers_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectArucoMarkers_GetResult_Response_result status(::as2_msgs::action::DetectArucoMarkers_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_DetectArucoMarkers_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_DetectArucoMarkers_FeedbackMessage_feedback
{
public:
  explicit Init_DetectArucoMarkers_FeedbackMessage_feedback(::as2_msgs::action::DetectArucoMarkers_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::DetectArucoMarkers_FeedbackMessage feedback(::as2_msgs::action::DetectArucoMarkers_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_FeedbackMessage msg_;
};

class Init_DetectArucoMarkers_FeedbackMessage_goal_id
{
public:
  Init_DetectArucoMarkers_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DetectArucoMarkers_FeedbackMessage_feedback goal_id(::as2_msgs::action::DetectArucoMarkers_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_DetectArucoMarkers_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::DetectArucoMarkers_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::DetectArucoMarkers_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_DetectArucoMarkers_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__DETECT_ARUCO_MARKERS__BUILDER_HPP_
