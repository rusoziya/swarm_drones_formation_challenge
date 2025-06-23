// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/PoseStampedWithIDArray.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POSE_STAMPED_WITH_ID_ARRAY__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__POSE_STAMPED_WITH_ID_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/pose_stamped_with_id_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_PoseStampedWithIDArray_poses
{
public:
  Init_PoseStampedWithIDArray_poses()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::msg::PoseStampedWithIDArray poses(::as2_msgs::msg::PoseStampedWithIDArray::_poses_type arg)
  {
    msg_.poses = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::PoseStampedWithIDArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::PoseStampedWithIDArray>()
{
  return as2_msgs::msg::builder::Init_PoseStampedWithIDArray_poses();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__POSE_STAMPED_WITH_ID_ARRAY__BUILDER_HPP_
