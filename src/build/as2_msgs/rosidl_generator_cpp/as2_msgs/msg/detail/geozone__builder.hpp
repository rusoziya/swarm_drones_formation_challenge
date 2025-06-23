// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/Geozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__GEOZONE__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__GEOZONE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/geozone__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_Geozone_z_down
{
public:
  explicit Init_Geozone_z_down(::as2_msgs::msg::Geozone & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::Geozone z_down(::as2_msgs::msg::Geozone::_z_down_type arg)
  {
    msg_.z_down = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

class Init_Geozone_z_up
{
public:
  explicit Init_Geozone_z_up(::as2_msgs::msg::Geozone & msg)
  : msg_(msg)
  {}
  Init_Geozone_z_down z_up(::as2_msgs::msg::Geozone::_z_up_type arg)
  {
    msg_.z_up = std::move(arg);
    return Init_Geozone_z_down(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

class Init_Geozone_polygon
{
public:
  explicit Init_Geozone_polygon(::as2_msgs::msg::Geozone & msg)
  : msg_(msg)
  {}
  Init_Geozone_z_up polygon(::as2_msgs::msg::Geozone::_polygon_type arg)
  {
    msg_.polygon = std::move(arg);
    return Init_Geozone_z_up(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

class Init_Geozone_data_type
{
public:
  explicit Init_Geozone_data_type(::as2_msgs::msg::Geozone & msg)
  : msg_(msg)
  {}
  Init_Geozone_polygon data_type(::as2_msgs::msg::Geozone::_data_type_type arg)
  {
    msg_.data_type = std::move(arg);
    return Init_Geozone_polygon(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

class Init_Geozone_type
{
public:
  explicit Init_Geozone_type(::as2_msgs::msg::Geozone & msg)
  : msg_(msg)
  {}
  Init_Geozone_data_type type(::as2_msgs::msg::Geozone::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_Geozone_data_type(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

class Init_Geozone_alert
{
public:
  explicit Init_Geozone_alert(::as2_msgs::msg::Geozone & msg)
  : msg_(msg)
  {}
  Init_Geozone_type alert(::as2_msgs::msg::Geozone::_alert_type arg)
  {
    msg_.alert = std::move(arg);
    return Init_Geozone_type(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

class Init_Geozone_id
{
public:
  Init_Geozone_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Geozone_alert id(::as2_msgs::msg::Geozone::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_Geozone_alert(msg_);
  }

private:
  ::as2_msgs::msg::Geozone msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::Geozone>()
{
  return as2_msgs::msg::builder::Init_Geozone_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__GEOZONE__BUILDER_HPP_
