// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/GetGeozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__GET_GEOZONE__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__GET_GEOZONE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/get_geozone__struct.hpp"
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
auto build<::as2_msgs::srv::GetGeozone_Request>()
{
  return ::as2_msgs::srv::GetGeozone_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_GetGeozone_Response_geozone_list
{
public:
  explicit Init_GetGeozone_Response_geozone_list(::as2_msgs::srv::GetGeozone_Response & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::GetGeozone_Response geozone_list(::as2_msgs::srv::GetGeozone_Response::_geozone_list_type arg)
  {
    msg_.geozone_list = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::GetGeozone_Response msg_;
};

class Init_GetGeozone_Response_success
{
public:
  Init_GetGeozone_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetGeozone_Response_geozone_list success(::as2_msgs::srv::GetGeozone_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetGeozone_Response_geozone_list(msg_);
  }

private:
  ::as2_msgs::srv::GetGeozone_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::GetGeozone_Response>()
{
  return as2_msgs::srv::builder::Init_GetGeozone_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__GET_GEOZONE__BUILDER_HPP_
