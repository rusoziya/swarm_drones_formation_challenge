// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/ControllerInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/controller_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_ControllerInfo_output_control_mode
{
public:
  explicit Init_ControllerInfo_output_control_mode(::as2_msgs::msg::ControllerInfo & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::ControllerInfo output_control_mode(::as2_msgs::msg::ControllerInfo::_output_control_mode_type arg)
  {
    msg_.output_control_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::ControllerInfo msg_;
};

class Init_ControllerInfo_input_control_mode
{
public:
  explicit Init_ControllerInfo_input_control_mode(::as2_msgs::msg::ControllerInfo & msg)
  : msg_(msg)
  {}
  Init_ControllerInfo_output_control_mode input_control_mode(::as2_msgs::msg::ControllerInfo::_input_control_mode_type arg)
  {
    msg_.input_control_mode = std::move(arg);
    return Init_ControllerInfo_output_control_mode(msg_);
  }

private:
  ::as2_msgs::msg::ControllerInfo msg_;
};

class Init_ControllerInfo_header
{
public:
  Init_ControllerInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ControllerInfo_input_control_mode header(::as2_msgs::msg::ControllerInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ControllerInfo_input_control_mode(msg_);
  }

private:
  ::as2_msgs::msg::ControllerInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::ControllerInfo>()
{
  return as2_msgs::msg::builder::Init_ControllerInfo_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__BUILDER_HPP_
