// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/PolygonList.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POLYGON_LIST__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__POLYGON_LIST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/polygon_list__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_PolygonList_polygons
{
public:
  Init_PolygonList_polygons()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::as2_msgs::msg::PolygonList polygons(::as2_msgs::msg::PolygonList::_polygons_type arg)
  {
    msg_.polygons = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::PolygonList msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::PolygonList>()
{
  return as2_msgs::msg::builder::Init_PolygonList_polygons();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__POLYGON_LIST__BUILDER_HPP_
