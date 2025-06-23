// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/PathToGeopath.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__PATH_TO_GEOPATH__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__PATH_TO_GEOPATH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/path_to_geopath__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_PathToGeopath_Request_path
{
public:
  Init_PathToGeopath_Request_path()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::PathToGeopath_Request path(::as2_msgs::srv::PathToGeopath_Request::_path_type arg)
  {
    msg_.path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::PathToGeopath_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::PathToGeopath_Request>()
{
  return as2_msgs::srv::builder::Init_PathToGeopath_Request_path();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_PathToGeopath_Response_geo_path
{
public:
  explicit Init_PathToGeopath_Response_geo_path(::as2_msgs::srv::PathToGeopath_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::PathToGeopath_Response geo_path(::as2_msgs::srv::PathToGeopath_Response::_geo_path_type arg)
  {
    msg_.geo_path = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::PathToGeopath_Response msg_;
};

class Init_PathToGeopath_Response_success
{
public:
  Init_PathToGeopath_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PathToGeopath_Response_geo_path success(::as2_msgs::srv::PathToGeopath_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PathToGeopath_Response_geo_path(msg_);
  }

private:
  ::as2_msgs::srv::PathToGeopath_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::PathToGeopath_Response>()
{
  return as2_msgs::srv::builder::Init_PathToGeopath_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__PATH_TO_GEOPATH__BUILDER_HPP_
