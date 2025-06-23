// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/TrajGenInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/traj_gen_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_TrajGenInfo_active_status
{
public:
  explicit Init_TrajGenInfo_active_status(::as2_msgs::msg::TrajGenInfo & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::TrajGenInfo active_status(::as2_msgs::msg::TrajGenInfo::_active_status_type arg)
  {
    msg_.active_status = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::TrajGenInfo msg_;
};

class Init_TrajGenInfo_node_status
{
public:
  explicit Init_TrajGenInfo_node_status(::as2_msgs::msg::TrajGenInfo & msg)
  : msg_(msg)
  {}
  Init_TrajGenInfo_active_status node_status(::as2_msgs::msg::TrajGenInfo::_node_status_type arg)
  {
    msg_.node_status = std::move(arg);
    return Init_TrajGenInfo_active_status(msg_);
  }

private:
  ::as2_msgs::msg::TrajGenInfo msg_;
};

class Init_TrajGenInfo_header
{
public:
  Init_TrajGenInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TrajGenInfo_node_status header(::as2_msgs::msg::TrajGenInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_TrajGenInfo_node_status(msg_);
  }

private:
  ::as2_msgs::msg::TrajGenInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::TrajGenInfo>()
{
  return as2_msgs::msg::builder::Init_TrajGenInfo_header();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__BUILDER_HPP_
