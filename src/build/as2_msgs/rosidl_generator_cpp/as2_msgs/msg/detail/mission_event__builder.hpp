// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/MissionEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__MISSION_EVENT__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__MISSION_EVENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/mission_event__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_MissionEvent_data
{
public:
  explicit Init_MissionEvent_data(::as2_msgs::msg::MissionEvent & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::MissionEvent data(::as2_msgs::msg::MissionEvent::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::MissionEvent msg_;
};

class Init_MissionEvent_header
{
public:
  Init_MissionEvent_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MissionEvent_data header(::as2_msgs::msg::MissionEvent::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_MissionEvent_data(msg_);
  }

private:
  ::as2_msgs::msg::MissionEvent msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::MissionEvent>()
{
  return as2_msgs::msg::builder::Init_MissionEvent_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__MISSION_EVENT__BUILDER_HPP_
