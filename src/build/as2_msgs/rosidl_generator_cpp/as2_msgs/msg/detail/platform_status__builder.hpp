// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/PlatformStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_STATUS__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/platform_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_PlatformStatus_state
{
public:
  Init_PlatformStatus_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::msg::PlatformStatus state(::as2_msgs::msg::PlatformStatus::_state_type arg)
  {
    msg_.state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::PlatformStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::PlatformStatus>()
{
  return as2_msgs::msg::builder::Init_PlatformStatus_state();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_STATUS__BUILDER_HPP_
