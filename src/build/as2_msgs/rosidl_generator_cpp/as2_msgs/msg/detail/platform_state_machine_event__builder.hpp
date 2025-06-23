// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/PlatformStateMachineEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/platform_state_machine_event__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_PlatformStateMachineEvent_event
{
public:
  Init_PlatformStateMachineEvent_event()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::msg::PlatformStateMachineEvent event(::as2_msgs::msg::PlatformStateMachineEvent::_event_type arg)
  {
    msg_.event = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::PlatformStateMachineEvent msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::PlatformStateMachineEvent>()
{
  return as2_msgs::msg::builder::Init_PlatformStateMachineEvent_event();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_STATE_MACHINE_EVENT__BUILDER_HPP_
