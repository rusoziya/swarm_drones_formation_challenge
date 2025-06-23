// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/TrajectorySetpoints.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJECTORY_SETPOINTS__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJECTORY_SETPOINTS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/trajectory_setpoints__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_TrajectorySetpoints_setpoints
{
public:
  explicit Init_TrajectorySetpoints_setpoints(::as2_msgs::msg::TrajectorySetpoints & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::TrajectorySetpoints setpoints(::as2_msgs::msg::TrajectorySetpoints::_setpoints_type arg)
  {
    msg_.setpoints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::TrajectorySetpoints msg_;
};

class Init_TrajectorySetpoints_header
{
public:
  Init_TrajectorySetpoints_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TrajectorySetpoints_setpoints header(::as2_msgs::msg::TrajectorySetpoints::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_TrajectorySetpoints_setpoints(msg_);
  }

private:
  ::as2_msgs::msg::TrajectorySetpoints msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::TrajectorySetpoints>()
{
  return as2_msgs::msg::builder::Init_TrajectorySetpoints_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__TRAJECTORY_SETPOINTS__BUILDER_HPP_
