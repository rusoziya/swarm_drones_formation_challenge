// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/PlatformInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/platform_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_PlatformInfo_current_control_mode
{
public:
  explicit Init_PlatformInfo_current_control_mode(::as2_msgs::msg::PlatformInfo & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::PlatformInfo current_control_mode(::as2_msgs::msg::PlatformInfo::_current_control_mode_type arg)
  {
    msg_.current_control_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::PlatformInfo msg_;
};

class Init_PlatformInfo_status
{
public:
  explicit Init_PlatformInfo_status(::as2_msgs::msg::PlatformInfo & msg)
  : msg_(msg)
  {}
  Init_PlatformInfo_current_control_mode status(::as2_msgs::msg::PlatformInfo::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_PlatformInfo_current_control_mode(msg_);
  }

private:
  ::as2_msgs::msg::PlatformInfo msg_;
};

class Init_PlatformInfo_offboard
{
public:
  explicit Init_PlatformInfo_offboard(::as2_msgs::msg::PlatformInfo & msg)
  : msg_(msg)
  {}
  Init_PlatformInfo_status offboard(::as2_msgs::msg::PlatformInfo::_offboard_type arg)
  {
    msg_.offboard = std::move(arg);
    return Init_PlatformInfo_status(msg_);
  }

private:
  ::as2_msgs::msg::PlatformInfo msg_;
};

class Init_PlatformInfo_armed
{
public:
  explicit Init_PlatformInfo_armed(::as2_msgs::msg::PlatformInfo & msg)
  : msg_(msg)
  {}
  Init_PlatformInfo_offboard armed(::as2_msgs::msg::PlatformInfo::_armed_type arg)
  {
    msg_.armed = std::move(arg);
    return Init_PlatformInfo_offboard(msg_);
  }

private:
  ::as2_msgs::msg::PlatformInfo msg_;
};

class Init_PlatformInfo_connected
{
public:
  explicit Init_PlatformInfo_connected(::as2_msgs::msg::PlatformInfo & msg)
  : msg_(msg)
  {}
  Init_PlatformInfo_armed connected(::as2_msgs::msg::PlatformInfo::_connected_type arg)
  {
    msg_.connected = std::move(arg);
    return Init_PlatformInfo_armed(msg_);
  }

private:
  ::as2_msgs::msg::PlatformInfo msg_;
};

class Init_PlatformInfo_header
{
public:
  Init_PlatformInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PlatformInfo_connected header(::as2_msgs::msg::PlatformInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PlatformInfo_connected(msg_);
  }

private:
  ::as2_msgs::msg::PlatformInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::PlatformInfo>()
{
  return as2_msgs::msg::builder::Init_PlatformInfo_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__BUILDER_HPP_
