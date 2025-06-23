// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/ListControlModes.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/list_control_modes__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::ListControlModes_Request>()
{
  return ::as2_msgs::srv::ListControlModes_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_ListControlModes_Response_control_modes
{
public:
  explicit Init_ListControlModes_Response_control_modes(::as2_msgs::srv::ListControlModes_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::ListControlModes_Response control_modes(::as2_msgs::srv::ListControlModes_Response::_control_modes_type arg)
  {
    msg_.control_modes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::ListControlModes_Response msg_;
};

class Init_ListControlModes_Response_source
{
public:
  Init_ListControlModes_Response_source()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ListControlModes_Response_control_modes source(::as2_msgs::srv::ListControlModes_Response::_source_type arg)
  {
    msg_.source = std::move(arg);
    return Init_ListControlModes_Response_control_modes(msg_);
  }

private:
  ::as2_msgs::srv::ListControlModes_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::ListControlModes_Response>()
{
  return as2_msgs::srv::builder::Init_ListControlModes_Response_source();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__BUILDER_HPP_
