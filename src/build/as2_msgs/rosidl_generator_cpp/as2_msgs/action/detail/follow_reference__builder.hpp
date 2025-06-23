// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:action/FollowReference.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__FOLLOW_REFERENCE__BUILDER_HPP_
#define AS2_MSGS__ACTION__DETAIL__FOLLOW_REFERENCE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/action/detail/follow_reference__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_Goal_max_speed_z
{
public:
  explicit Init_FollowReference_Goal_max_speed_z(::as2_msgs::action::FollowReference_Goal & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowReference_Goal max_speed_z(::as2_msgs::action::FollowReference_Goal::_max_speed_z_type arg)
  {
    msg_.max_speed_z = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Goal msg_;
};

class Init_FollowReference_Goal_max_speed_y
{
public:
  explicit Init_FollowReference_Goal_max_speed_y(::as2_msgs::action::FollowReference_Goal & msg)
  : msg_(msg)
  {}
  Init_FollowReference_Goal_max_speed_z max_speed_y(::as2_msgs::action::FollowReference_Goal::_max_speed_y_type arg)
  {
    msg_.max_speed_y = std::move(arg);
    return Init_FollowReference_Goal_max_speed_z(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Goal msg_;
};

class Init_FollowReference_Goal_max_speed_x
{
public:
  explicit Init_FollowReference_Goal_max_speed_x(::as2_msgs::action::FollowReference_Goal & msg)
  : msg_(msg)
  {}
  Init_FollowReference_Goal_max_speed_y max_speed_x(::as2_msgs::action::FollowReference_Goal::_max_speed_x_type arg)
  {
    msg_.max_speed_x = std::move(arg);
    return Init_FollowReference_Goal_max_speed_y(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Goal msg_;
};

class Init_FollowReference_Goal_target_pose
{
public:
  explicit Init_FollowReference_Goal_target_pose(::as2_msgs::action::FollowReference_Goal & msg)
  : msg_(msg)
  {}
  Init_FollowReference_Goal_max_speed_x target_pose(::as2_msgs::action::FollowReference_Goal::_target_pose_type arg)
  {
    msg_.target_pose = std::move(arg);
    return Init_FollowReference_Goal_max_speed_x(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Goal msg_;
};

class Init_FollowReference_Goal_yaw
{
public:
  Init_FollowReference_Goal_yaw()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowReference_Goal_target_pose yaw(::as2_msgs::action::FollowReference_Goal::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_FollowReference_Goal_target_pose(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_Goal>()
{
  return as2_msgs::action::builder::Init_FollowReference_Goal_yaw();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_Result_follow_reference_success
{
public:
  Init_FollowReference_Result_follow_reference_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::FollowReference_Result follow_reference_success(::as2_msgs::action::FollowReference_Result::_follow_reference_success_type arg)
  {
    msg_.follow_reference_success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_Result>()
{
  return as2_msgs::action::builder::Init_FollowReference_Result_follow_reference_success();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_Feedback_actual_distance_to_goal
{
public:
  explicit Init_FollowReference_Feedback_actual_distance_to_goal(::as2_msgs::action::FollowReference_Feedback & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowReference_Feedback actual_distance_to_goal(::as2_msgs::action::FollowReference_Feedback::_actual_distance_to_goal_type arg)
  {
    msg_.actual_distance_to_goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Feedback msg_;
};

class Init_FollowReference_Feedback_actual_speed
{
public:
  Init_FollowReference_Feedback_actual_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowReference_Feedback_actual_distance_to_goal actual_speed(::as2_msgs::action::FollowReference_Feedback::_actual_speed_type arg)
  {
    msg_.actual_speed = std::move(arg);
    return Init_FollowReference_Feedback_actual_distance_to_goal(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_Feedback msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_Feedback>()
{
  return as2_msgs::action::builder::Init_FollowReference_Feedback_actual_speed();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_SendGoal_Request_goal
{
public:
  explicit Init_FollowReference_SendGoal_Request_goal(::as2_msgs::action::FollowReference_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowReference_SendGoal_Request goal(::as2_msgs::action::FollowReference_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_SendGoal_Request msg_;
};

class Init_FollowReference_SendGoal_Request_goal_id
{
public:
  Init_FollowReference_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowReference_SendGoal_Request_goal goal_id(::as2_msgs::action::FollowReference_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_FollowReference_SendGoal_Request_goal(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_SendGoal_Request>()
{
  return as2_msgs::action::builder::Init_FollowReference_SendGoal_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_SendGoal_Response_stamp
{
public:
  explicit Init_FollowReference_SendGoal_Response_stamp(::as2_msgs::action::FollowReference_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowReference_SendGoal_Response stamp(::as2_msgs::action::FollowReference_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_SendGoal_Response msg_;
};

class Init_FollowReference_SendGoal_Response_accepted
{
public:
  Init_FollowReference_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowReference_SendGoal_Response_stamp accepted(::as2_msgs::action::FollowReference_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_FollowReference_SendGoal_Response_stamp(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_SendGoal_Response>()
{
  return as2_msgs::action::builder::Init_FollowReference_SendGoal_Response_accepted();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_GetResult_Request_goal_id
{
public:
  Init_FollowReference_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::action::FollowReference_GetResult_Request goal_id(::as2_msgs::action::FollowReference_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_GetResult_Request>()
{
  return as2_msgs::action::builder::Init_FollowReference_GetResult_Request_goal_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_GetResult_Response_result
{
public:
  explicit Init_FollowReference_GetResult_Response_result(::as2_msgs::action::FollowReference_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowReference_GetResult_Response result(::as2_msgs::action::FollowReference_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_GetResult_Response msg_;
};

class Init_FollowReference_GetResult_Response_status
{
public:
  Init_FollowReference_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowReference_GetResult_Response_result status(::as2_msgs::action::FollowReference_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_FollowReference_GetResult_Response_result(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_GetResult_Response>()
{
  return as2_msgs::action::builder::Init_FollowReference_GetResult_Response_status();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace action
{

namespace builder
{

class Init_FollowReference_FeedbackMessage_feedback
{
public:
  explicit Init_FollowReference_FeedbackMessage_feedback(::as2_msgs::action::FollowReference_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::as2_msgs::action::FollowReference_FeedbackMessage feedback(::as2_msgs::action::FollowReference_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_FeedbackMessage msg_;
};

class Init_FollowReference_FeedbackMessage_goal_id
{
public:
  Init_FollowReference_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowReference_FeedbackMessage_feedback goal_id(::as2_msgs::action::FollowReference_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_FollowReference_FeedbackMessage_feedback(msg_);
  }

private:
  ::as2_msgs::action::FollowReference_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::action::FollowReference_FeedbackMessage>()
{
  return as2_msgs::action::builder::Init_FollowReference_FeedbackMessage_goal_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__FOLLOW_REFERENCE__BUILDER_HPP_
