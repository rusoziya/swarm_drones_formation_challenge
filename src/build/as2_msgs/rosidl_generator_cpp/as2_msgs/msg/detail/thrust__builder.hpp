// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/Thrust.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__THRUST__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__THRUST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/thrust__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_Thrust_thrust_normalized
{
public:
  explicit Init_Thrust_thrust_normalized(::as2_msgs::msg::Thrust & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::Thrust thrust_normalized(::as2_msgs::msg::Thrust::_thrust_normalized_type arg)
  {
    msg_.thrust_normalized = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::Thrust msg_;
};

class Init_Thrust_thrust
{
public:
  explicit Init_Thrust_thrust(::as2_msgs::msg::Thrust & msg)
  : msg_(msg)
  {}
  Init_Thrust_thrust_normalized thrust(::as2_msgs::msg::Thrust::_thrust_type arg)
  {
    msg_.thrust = std::move(arg);
    return Init_Thrust_thrust_normalized(msg_);
  }

private:
  ::as2_msgs::msg::Thrust msg_;
};

class Init_Thrust_header
{
public:
  Init_Thrust_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Thrust_thrust header(::as2_msgs::msg::Thrust::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Thrust_thrust(msg_);
  }

private:
  ::as2_msgs::msg::Thrust msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::Thrust>()
{
  return as2_msgs::msg::builder::Init_Thrust_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__THRUST__BUILDER_HPP_
