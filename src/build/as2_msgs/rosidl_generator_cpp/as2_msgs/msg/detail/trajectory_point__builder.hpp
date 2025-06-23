// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/TrajectoryPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/trajectory_point__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_TrajectoryPoint_yaw_angle
{
public:
  explicit Init_TrajectoryPoint_yaw_angle(::as2_msgs::msg::TrajectoryPoint & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::TrajectoryPoint yaw_angle(::as2_msgs::msg::TrajectoryPoint::_yaw_angle_type arg)
  {
    msg_.yaw_angle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::TrajectoryPoint msg_;
};

class Init_TrajectoryPoint_acceleration
{
public:
  explicit Init_TrajectoryPoint_acceleration(::as2_msgs::msg::TrajectoryPoint & msg)
  : msg_(msg)
  {}
  Init_TrajectoryPoint_yaw_angle acceleration(::as2_msgs::msg::TrajectoryPoint::_acceleration_type arg)
  {
    msg_.acceleration = std::move(arg);
    return Init_TrajectoryPoint_yaw_angle(msg_);
  }

private:
  ::as2_msgs::msg::TrajectoryPoint msg_;
};

class Init_TrajectoryPoint_twist
{
public:
  explicit Init_TrajectoryPoint_twist(::as2_msgs::msg::TrajectoryPoint & msg)
  : msg_(msg)
  {}
  Init_TrajectoryPoint_acceleration twist(::as2_msgs::msg::TrajectoryPoint::_twist_type arg)
  {
    msg_.twist = std::move(arg);
    return Init_TrajectoryPoint_acceleration(msg_);
  }

private:
  ::as2_msgs::msg::TrajectoryPoint msg_;
};

class Init_TrajectoryPoint_position
{
public:
  Init_TrajectoryPoint_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TrajectoryPoint_twist position(::as2_msgs::msg::TrajectoryPoint::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_TrajectoryPoint_twist(msg_);
  }

private:
  ::as2_msgs::msg::TrajectoryPoint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::TrajectoryPoint>()
{
  return as2_msgs::msg::builder::Init_TrajectoryPoint_position();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__BUILDER_HPP_
