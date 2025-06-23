// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/SetControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/set_control_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetControlMode_Request_control_mode
{
public:
  Init_SetControlMode_Request_control_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetControlMode_Request control_mode(::as2_msgs::srv::SetControlMode_Request::_control_mode_type arg)
  {
    msg_.control_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetControlMode_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetControlMode_Request>()
{
  return as2_msgs::srv::builder::Init_SetControlMode_Request_control_mode();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetControlMode_Response_success
{
public:
  Init_SetControlMode_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetControlMode_Response success(::as2_msgs::srv::SetControlMode_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetControlMode_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetControlMode_Response>()
{
  return as2_msgs::srv::builder::Init_SetControlMode_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__BUILDER_HPP_
