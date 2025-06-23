// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/GeopathToPath.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__GEOPATH_TO_PATH__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__GEOPATH_TO_PATH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/geopath_to_path__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_GeopathToPath_Request_geo_path
{
public:
  Init_GeopathToPath_Request_geo_path()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::GeopathToPath_Request geo_path(::as2_msgs::srv::GeopathToPath_Request::_geo_path_type arg)
  {
    msg_.geo_path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::GeopathToPath_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::GeopathToPath_Request>()
{
  return as2_msgs::srv::builder::Init_GeopathToPath_Request_geo_path();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_GeopathToPath_Response_path
{
public:
  explicit Init_GeopathToPath_Response_path(::as2_msgs::srv::GeopathToPath_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::GeopathToPath_Response path(::as2_msgs::srv::GeopathToPath_Response::_path_type arg)
  {
    msg_.path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::GeopathToPath_Response msg_;
};

class Init_GeopathToPath_Response_success
{
public:
  Init_GeopathToPath_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GeopathToPath_Response_path success(::as2_msgs::srv::GeopathToPath_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GeopathToPath_Response_path(msg_);
  }

private:
  ::as2_msgs::srv::GeopathToPath_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::GeopathToPath_Response>()
{
  return as2_msgs::srv::builder::Init_GeopathToPath_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__GEOPATH_TO_PATH__BUILDER_HPP_
