// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/FollowTargetInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/follow_target_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_FollowTargetInfo_follow_mode
{
public:
  explicit Init_FollowTargetInfo_follow_mode(::as2_msgs::msg::FollowTargetInfo & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::FollowTargetInfo follow_mode(::as2_msgs::msg::FollowTargetInfo::_follow_mode_type arg)
  {
    msg_.follow_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::FollowTargetInfo msg_;
};

class Init_FollowTargetInfo_follow_status
{
public:
  explicit Init_FollowTargetInfo_follow_status(::as2_msgs::msg::FollowTargetInfo & msg)
  : msg_(msg)
  {}
  Init_FollowTargetInfo_follow_mode follow_status(::as2_msgs::msg::FollowTargetInfo::_follow_status_type arg)
  {
    msg_.follow_status = std::move(arg);
    return Init_FollowTargetInfo_follow_mode(msg_);
  }

private:
  ::as2_msgs::msg::FollowTargetInfo msg_;
};

class Init_FollowTargetInfo_header
{
public:
  Init_FollowTargetInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FollowTargetInfo_follow_status header(::as2_msgs::msg::FollowTargetInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_FollowTargetInfo_follow_status(msg_);
  }

private:
  ::as2_msgs::msg::FollowTargetInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::FollowTargetInfo>()
{
  return as2_msgs::msg::builder::Init_FollowTargetInfo_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__BUILDER_HPP_
