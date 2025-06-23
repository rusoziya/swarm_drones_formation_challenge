// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/YawMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__YAW_MODE__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__YAW_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/yaw_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_YawMode_angle
{
public:
  explicit Init_YawMode_angle(::as2_msgs::msg::YawMode & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::YawMode angle(::as2_msgs::msg::YawMode::_angle_type arg)
  {
    msg_.angle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::YawMode msg_;
};

class Init_YawMode_mode
{
public:
  Init_YawMode_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_YawMode_angle mode(::as2_msgs::msg::YawMode::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_YawMode_angle(msg_);
  }

private:
  ::as2_msgs::msg::YawMode msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::YawMode>()
{
  return as2_msgs::msg::builder::Init_YawMode_mode();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__YAW_MODE__BUILDER_HPP_
