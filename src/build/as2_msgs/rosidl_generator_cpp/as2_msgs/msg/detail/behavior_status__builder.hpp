// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/BehaviorStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/behavior_status__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_BehaviorStatus_status
{
public:
  Init_BehaviorStatus_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::msg::BehaviorStatus status(::as2_msgs::msg::BehaviorStatus::_status_type arg)
  {
    msg_.status = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::BehaviorStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::BehaviorStatus>()
{
  return as2_msgs::msg::builder::Init_BehaviorStatus_status();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__BUILDER_HPP_
