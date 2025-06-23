// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/Speed.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__SPEED__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__SPEED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/speed__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_Speed_speed
{
public:
  explicit Init_Speed_speed(::as2_msgs::msg::Speed & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::Speed speed(::as2_msgs::msg::Speed::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::Speed msg_;
};

class Init_Speed_header
{
public:
  Init_Speed_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Speed_speed header(::as2_msgs::msg::Speed::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Speed_speed(msg_);
  }

private:
  ::as2_msgs::msg::Speed msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::Speed>()
{
  return as2_msgs::msg::builder::Init_Speed_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__SPEED__BUILDER_HPP_
