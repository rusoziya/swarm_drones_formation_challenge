// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/Acro.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ACRO__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__ACRO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/acro__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_Acro_thrust
{
public:
  explicit Init_Acro_thrust(::as2_msgs::msg::Acro & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::Acro thrust(::as2_msgs::msg::Acro::_thrust_type arg)
  {
    msg_.thrust = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::Acro msg_;
};

class Init_Acro_angular_rates
{
public:
  explicit Init_Acro_angular_rates(::as2_msgs::msg::Acro & msg)
  : msg_(msg)
  {}
  Init_Acro_thrust angular_rates(::as2_msgs::msg::Acro::_angular_rates_type arg)
  {
    msg_.angular_rates = std::move(arg);
    return Init_Acro_thrust(msg_);
  }

private:
  ::as2_msgs::msg::Acro msg_;
};

class Init_Acro_header
{
public:
  Init_Acro_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Acro_angular_rates header(::as2_msgs::msg::Acro::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Acro_angular_rates(msg_);
  }

private:
  ::as2_msgs::msg::Acro msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::Acro>()
{
  return as2_msgs::msg::builder::Init_Acro_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__ACRO__BUILDER_HPP_
