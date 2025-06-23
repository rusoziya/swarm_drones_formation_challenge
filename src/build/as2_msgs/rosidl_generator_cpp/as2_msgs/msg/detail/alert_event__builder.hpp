// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/AlertEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ALERT_EVENT__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__ALERT_EVENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/alert_event__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_AlertEvent_description
{
public:
  explicit Init_AlertEvent_description(::as2_msgs::msg::AlertEvent & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::AlertEvent description(::as2_msgs::msg::AlertEvent::_description_type arg)
  {
    msg_.description = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::AlertEvent msg_;
};

class Init_AlertEvent_alert
{
public:
  Init_AlertEvent_alert()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AlertEvent_description alert(::as2_msgs::msg::AlertEvent::_alert_type arg)
  {
    msg_.alert = std::move(arg);
    return Init_AlertEvent_description(msg_);
  }

private:
  ::as2_msgs::msg::AlertEvent msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::AlertEvent>()
{
  return as2_msgs::msg::builder::Init_AlertEvent_alert();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__ALERT_EVENT__BUILDER_HPP_
