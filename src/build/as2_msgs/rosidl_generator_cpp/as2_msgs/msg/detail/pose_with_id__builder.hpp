// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/PoseWithID.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/pose_with_id__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_PoseWithID_pose
{
public:
  explicit Init_PoseWithID_pose(::as2_msgs::msg::PoseWithID & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::PoseWithID pose(::as2_msgs::msg::PoseWithID::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::PoseWithID msg_;
};

class Init_PoseWithID_id
{
public:
  Init_PoseWithID_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PoseWithID_pose id(::as2_msgs::msg::PoseWithID::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_PoseWithID_pose(msg_);
  }

private:
  ::as2_msgs::msg::PoseWithID msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::PoseWithID>()
{
  return as2_msgs::msg::builder::Init_PoseWithID_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__BUILDER_HPP_
