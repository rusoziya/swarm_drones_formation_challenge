// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/AddStaticTransform.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/add_static_transform__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_AddStaticTransform_Request_transform
{
public:
  explicit Init_AddStaticTransform_Request_transform(::as2_msgs::srv::AddStaticTransform_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::AddStaticTransform_Request transform(::as2_msgs::srv::AddStaticTransform_Request::_transform_type arg)
  {
    msg_.transform = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransform_Request msg_;
};

class Init_AddStaticTransform_Request_child_frame_id
{
public:
  explicit Init_AddStaticTransform_Request_child_frame_id(::as2_msgs::srv::AddStaticTransform_Request & msg)
  : msg_(msg)
  {}
  Init_AddStaticTransform_Request_transform child_frame_id(::as2_msgs::srv::AddStaticTransform_Request::_child_frame_id_type arg)
  {
    msg_.child_frame_id = std::move(arg);
    return Init_AddStaticTransform_Request_transform(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransform_Request msg_;
};

class Init_AddStaticTransform_Request_frame_id
{
public:
  Init_AddStaticTransform_Request_frame_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AddStaticTransform_Request_child_frame_id frame_id(::as2_msgs::srv::AddStaticTransform_Request::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_AddStaticTransform_Request_child_frame_id(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransform_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::AddStaticTransform_Request>()
{
  return as2_msgs::srv::builder::Init_AddStaticTransform_Request_frame_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_AddStaticTransform_Response_success
{
public:
  Init_AddStaticTransform_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::AddStaticTransform_Response success(::as2_msgs::srv::AddStaticTransform_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransform_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::AddStaticTransform_Response>()
{
  return as2_msgs::srv::builder::Init_AddStaticTransform_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM__BUILDER_HPP_
