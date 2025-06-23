// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/GetOrigin.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__GET_ORIGIN__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__GET_ORIGIN__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/get_origin__struct.hpp"
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
auto build<::as2_msgs::srv::GetOrigin_Request>()
{
  return ::as2_msgs::srv::GetOrigin_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_GetOrigin_Response_origin
{
public:
  explicit Init_GetOrigin_Response_origin(::as2_msgs::srv::GetOrigin_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::GetOrigin_Response origin(::as2_msgs::srv::GetOrigin_Response::_origin_type arg)
  {
    msg_.origin = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::GetOrigin_Response msg_;
};

class Init_GetOrigin_Response_success
{
public:
  Init_GetOrigin_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetOrigin_Response_origin success(::as2_msgs::srv::GetOrigin_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetOrigin_Response_origin(msg_);
  }

private:
  ::as2_msgs::srv::GetOrigin_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::GetOrigin_Response>()
{
  return as2_msgs::srv::builder::Init_GetOrigin_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__GET_ORIGIN__BUILDER_HPP_
