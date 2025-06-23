// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/SetGeozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_GEOZONE__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_GEOZONE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/set_geozone__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetGeozone_Request_geozone
{
public:
  Init_SetGeozone_Request_geozone()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetGeozone_Request geozone(::as2_msgs::srv::SetGeozone_Request::_geozone_type arg)
  {
    msg_.geozone = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetGeozone_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetGeozone_Request>()
{
  return as2_msgs::srv::builder::Init_SetGeozone_Request_geozone();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_SetGeozone_Response_success
{
public:
  Init_SetGeozone_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::SetGeozone_Response success(::as2_msgs::srv::SetGeozone_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::SetGeozone_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::SetGeozone_Response>()
{
  return as2_msgs::srv::builder::Init_SetGeozone_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_GEOZONE__BUILDER_HPP_
