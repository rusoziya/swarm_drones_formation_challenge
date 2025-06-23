// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/ControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROL_MODE__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__CONTROL_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/control_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_ControlMode_reference_frame
{
public:
  explicit Init_ControlMode_reference_frame(::as2_msgs::msg::ControlMode & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::ControlMode reference_frame(::as2_msgs::msg::ControlMode::_reference_frame_type arg)
  {
    msg_.reference_frame = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::ControlMode msg_;
};

class Init_ControlMode_control_mode
{
public:
  explicit Init_ControlMode_control_mode(::as2_msgs::msg::ControlMode & msg)
  : msg_(msg)
  {}
  Init_ControlMode_reference_frame control_mode(::as2_msgs::msg::ControlMode::_control_mode_type arg)
  {
    msg_.control_mode = std::move(arg);
    return Init_ControlMode_reference_frame(msg_);
  }

private:
  ::as2_msgs::msg::ControlMode msg_;
};

class Init_ControlMode_yaw_mode
{
public:
  explicit Init_ControlMode_yaw_mode(::as2_msgs::msg::ControlMode & msg)
  : msg_(msg)
  {}
  Init_ControlMode_control_mode yaw_mode(::as2_msgs::msg::ControlMode::_yaw_mode_type arg)
  {
    msg_.yaw_mode = std::move(arg);
    return Init_ControlMode_control_mode(msg_);
  }

private:
  ::as2_msgs::msg::ControlMode msg_;
};

class Init_ControlMode_header
{
public:
  Init_ControlMode_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ControlMode_yaw_mode header(::as2_msgs::msg::ControlMode::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ControlMode_yaw_mode(msg_);
  }

private:
  ::as2_msgs::msg::ControlMode msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::ControlMode>()
{
  return as2_msgs::msg::builder::Init_ControlMode_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__CONTROL_MODE__BUILDER_HPP_
