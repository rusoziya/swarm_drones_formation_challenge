// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/SetSpeed.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_SPEED__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_SPEED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/set_speed__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetSpeed_Request_speed
{
public:
  Init_SetSpeed_Request_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetSpeed_Request speed(::as2_msgs::srv::SetSpeed_Request::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetSpeed_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetSpeed_Request>()
{
  return as2_msgs::srv::builder::Init_SetSpeed_Request_speed();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetSpeed_Response_success
{
public:
  Init_SetSpeed_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetSpeed_Response success(::as2_msgs::srv::SetSpeed_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetSpeed_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetSpeed_Response>()
{
  return as2_msgs::srv::builder::Init_SetSpeed_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_SPEED__BUILDER_HPP_
