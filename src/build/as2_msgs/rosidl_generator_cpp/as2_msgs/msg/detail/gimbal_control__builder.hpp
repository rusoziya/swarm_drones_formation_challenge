// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/GimbalControl.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/gimbal_control__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_GimbalControl_target
{
public:
  explicit Init_GimbalControl_target(::as2_msgs::msg::GimbalControl & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::GimbalControl target(::as2_msgs::msg::GimbalControl::_target_type arg)
  {
    msg_.target = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::GimbalControl msg_;
};

class Init_GimbalControl_control_mode
{
public:
  Init_GimbalControl_control_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GimbalControl_target control_mode(::as2_msgs::msg::GimbalControl::_control_mode_type arg)
  {
    msg_.control_mode = std::move(arg);
    return Init_GimbalControl_target(msg_);
  }

private:
  ::as2_msgs::msg::GimbalControl msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::GimbalControl>()
{
  return as2_msgs::msg::builder::Init_GimbalControl_control_mode();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__GIMBAL_CONTROL__BUILDER_HPP_
