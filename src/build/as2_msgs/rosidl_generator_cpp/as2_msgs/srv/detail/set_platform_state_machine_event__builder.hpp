// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/SetPlatformStateMachineEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/set_platform_state_machine_event__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetPlatformStateMachineEvent_Request_event
{
public:
  Init_SetPlatformStateMachineEvent_Request_event()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetPlatformStateMachineEvent_Request event(::as2_msgs::srv::SetPlatformStateMachineEvent_Request::_event_type arg)
  {
    msg_.event = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetPlatformStateMachineEvent_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetPlatformStateMachineEvent_Request>()
{
  return as2_msgs::srv::builder::Init_SetPlatformStateMachineEvent_Request_event();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetPlatformStateMachineEvent_Response_current_state
{
public:
  explicit Init_SetPlatformStateMachineEvent_Response_current_state(::as2_msgs::srv::SetPlatformStateMachineEvent_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::SetPlatformStateMachineEvent_Response current_state(::as2_msgs::srv::SetPlatformStateMachineEvent_Response::_current_state_type arg)
  {
    msg_.current_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetPlatformStateMachineEvent_Response msg_;
};

class Init_SetPlatformStateMachineEvent_Response_success
{
public:
  Init_SetPlatformStateMachineEvent_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetPlatformStateMachineEvent_Response_current_state success(::as2_msgs::srv::SetPlatformStateMachineEvent_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetPlatformStateMachineEvent_Response_current_state(msg_);
  }

private:
  ::as2_msgs::srv::SetPlatformStateMachineEvent_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetPlatformStateMachineEvent_Response>()
{
  return as2_msgs::srv::builder::Init_SetPlatformStateMachineEvent_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__BUILDER_HPP_
