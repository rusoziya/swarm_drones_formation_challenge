// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:srv/AddStaticTransformGps.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__BUILDER_HPP_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/srv/detail/add_static_transform_gps__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_AddStaticTransformGps_Request_bank
{
public:
  explicit Init_AddStaticTransformGps_Request_bank(::as2_msgs::srv::AddStaticTransformGps_Request & msg)
  : msg_(msg)
  {}
  ::as2_msgs::srv::AddStaticTransformGps_Request bank(::as2_msgs::srv::AddStaticTransformGps_Request::_bank_type arg)
  {
    msg_.bank = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Request msg_;
};

class Init_AddStaticTransformGps_Request_elevation
{
public:
  explicit Init_AddStaticTransformGps_Request_elevation(::as2_msgs::srv::AddStaticTransformGps_Request & msg)
  : msg_(msg)
  {}
  Init_AddStaticTransformGps_Request_bank elevation(::as2_msgs::srv::AddStaticTransformGps_Request::_elevation_type arg)
  {
    msg_.elevation = std::move(arg);
    return Init_AddStaticTransformGps_Request_bank(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Request msg_;
};

class Init_AddStaticTransformGps_Request_azimuth
{
public:
  explicit Init_AddStaticTransformGps_Request_azimuth(::as2_msgs::srv::AddStaticTransformGps_Request & msg)
  : msg_(msg)
  {}
  Init_AddStaticTransformGps_Request_elevation azimuth(::as2_msgs::srv::AddStaticTransformGps_Request::_azimuth_type arg)
  {
    msg_.azimuth = std::move(arg);
    return Init_AddStaticTransformGps_Request_elevation(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Request msg_;
};

class Init_AddStaticTransformGps_Request_gps_position
{
public:
  explicit Init_AddStaticTransformGps_Request_gps_position(::as2_msgs::srv::AddStaticTransformGps_Request & msg)
  : msg_(msg)
  {}
  Init_AddStaticTransformGps_Request_azimuth gps_position(::as2_msgs::srv::AddStaticTransformGps_Request::_gps_position_type arg)
  {
    msg_.gps_position = std::move(arg);
    return Init_AddStaticTransformGps_Request_azimuth(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Request msg_;
};

class Init_AddStaticTransformGps_Request_child_frame_id
{
public:
  explicit Init_AddStaticTransformGps_Request_child_frame_id(::as2_msgs::srv::AddStaticTransformGps_Request & msg)
  : msg_(msg)
  {}
  Init_AddStaticTransformGps_Request_gps_position child_frame_id(::as2_msgs::srv::AddStaticTransformGps_Request::_child_frame_id_type arg)
  {
    msg_.child_frame_id = std::move(arg);
    return Init_AddStaticTransformGps_Request_gps_position(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Request msg_;
};

class Init_AddStaticTransformGps_Request_frame_id
{
public:
  Init_AddStaticTransformGps_Request_frame_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AddStaticTransformGps_Request_child_frame_id frame_id(::as2_msgs::srv::AddStaticTransformGps_Request::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_AddStaticTransformGps_Request_child_frame_id(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::AddStaticTransformGps_Request>()
{
  return as2_msgs::srv::builder::Init_AddStaticTransformGps_Request_frame_id();
}

}  // namespace as2_msgs


namespace as2_msgs
{

namespace srv
{

namespace builder
{

class Init_AddStaticTransformGps_Response_success
{
public:
  Init_AddStaticTransformGps_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::srv::AddStaticTransformGps_Response success(::as2_msgs::srv::AddStaticTransformGps_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::srv::AddStaticTransformGps_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::srv::AddStaticTransformGps_Response>()
{
  return as2_msgs::srv::builder::Init_AddStaticTransformGps_Response_success();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__BUILDER_HPP_
