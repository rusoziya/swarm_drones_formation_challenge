// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/DynamicFollower.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__DYNAMIC_FOLLOWER__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__DYNAMIC_FOLLOWER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/dynamic_follower__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_DynamicFollower_Request_speed_limit
{
public:
  explicit Init_DynamicFollower_Request_speed_limit(::as2_msgs::srv::DynamicFollower_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::DynamicFollower_Request speed_limit(::as2_msgs::srv::DynamicFollower_Request::_speed_limit_type arg)
  {
    msg_.speed_limit = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::DynamicFollower_Request msg_;
};

class Init_DynamicFollower_Request_enable
{
public:
  Init_DynamicFollower_Request_enable()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DynamicFollower_Request_speed_limit enable(::as2_msgs::srv::DynamicFollower_Request::_enable_type arg)
  {
    msg_.enable = std::move(arg);
    return Init_DynamicFollower_Request_speed_limit(msg_);
  }

private:
  ::as2_msgs::srv::DynamicFollower_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::DynamicFollower_Request>()
{
  return as2_msgs::srv::builder::Init_DynamicFollower_Request_enable();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_DynamicFollower_Response_success
{
public:
  Init_DynamicFollower_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::DynamicFollower_Response success(::as2_msgs::srv::DynamicFollower_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::DynamicFollower_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::DynamicFollower_Response>()
{
  return as2_msgs::srv::builder::Init_DynamicFollower_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__DYNAMIC_FOLLOWER__BUILDER_HPP_
