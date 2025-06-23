// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from as2_msgs:msg/MissionUpdate.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__BUILDER_HPP_
#define AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "as2_msgs/msg/detail/mission_update__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace as2_msgs
{

namespace msg
{

namespace builder
{

class Init_MissionUpdate_mission
{
public:
  explicit Init_MissionUpdate_mission(::as2_msgs::msg::MissionUpdate & msg)
  : msg_(msg)
  {}
  ::as2_msgs::msg::MissionUpdate mission(::as2_msgs::msg::MissionUpdate::_mission_type arg)
  {
    msg_.mission = std::move(arg);
    return std::move(msg_);
  }

private:
  ::as2_msgs::msg::MissionUpdate msg_;
};

class Init_MissionUpdate_action
{
public:
  explicit Init_MissionUpdate_action(::as2_msgs::msg::MissionUpdate & msg)
  : msg_(msg)
  {}
  Init_MissionUpdate_mission action(::as2_msgs::msg::MissionUpdate::_action_type arg)
  {
    msg_.action = std::move(arg);
    return Init_MissionUpdate_mission(msg_);
  }

private:
  ::as2_msgs::msg::MissionUpdate msg_;
};

class Init_MissionUpdate_item_id
{
public:
  explicit Init_MissionUpdate_item_id(::as2_msgs::msg::MissionUpdate & msg)
  : msg_(msg)
  {}
  Init_MissionUpdate_action item_id(::as2_msgs::msg::MissionUpdate::_item_id_type arg)
  {
    msg_.item_id = std::move(arg);
    return Init_MissionUpdate_action(msg_);
  }

private:
  ::as2_msgs::msg::MissionUpdate msg_;
};

class Init_MissionUpdate_mission_id
{
public:
  explicit Init_MissionUpdate_mission_id(::as2_msgs::msg::MissionUpdate & msg)
  : msg_(msg)
  {}
  Init_MissionUpdate_item_id mission_id(::as2_msgs::msg::MissionUpdate::_mission_id_type arg)
  {
    msg_.mission_id = std::move(arg);
    return Init_MissionUpdate_item_id(msg_);
  }

private:
  ::as2_msgs::msg::MissionUpdate msg_;
};

class Init_MissionUpdate_drone_id
{
public:
  Init_MissionUpdate_drone_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MissionUpdate_mission_id drone_id(::as2_msgs::msg::MissionUpdate::_drone_id_type arg)
  {
    msg_.drone_id = std::move(arg);
    return Init_MissionUpdate_mission_id(msg_);
  }

private:
  ::as2_msgs::msg::MissionUpdate msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::as2_msgs::msg::MissionUpdate>()
{
  return as2_msgs::msg::builder::Init_MissionUpdate_drone_id();
}

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__BUILDER_HPP_
