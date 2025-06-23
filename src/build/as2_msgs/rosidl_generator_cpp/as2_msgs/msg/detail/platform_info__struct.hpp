// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/PlatformInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'status'
#include "as2_msgs/msg/detail/platform_status__struct.hpp"
// Member 'current_control_mode'
#include "as2_msgs/msg/detail/control_mode__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__PlatformInfo __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__PlatformInfo __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PlatformInfo_
{
  using Type = PlatformInfo_<ContainerAllocator>;

  explicit PlatformInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    status(_init),
    current_control_mode(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->connected = false;
      this->armed = false;
      this->offboard = false;
    }
  }

  explicit PlatformInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    status(_alloc, _init),
    current_control_mode(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->connected = false;
      this->armed = false;
      this->offboard = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _connected_type =
    bool;
  _connected_type connected;
  using _armed_type =
    bool;
  _armed_type armed;
  using _offboard_type =
    bool;
  _offboard_type offboard;
  using _status_type =
    as2_msgs::msg::PlatformStatus_<ContainerAllocator>;
  _status_type status;
  using _current_control_mode_type =
    as2_msgs::msg::ControlMode_<ContainerAllocator>;
  _current_control_mode_type current_control_mode;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__connected(
    const bool & _arg)
  {
    this->connected = _arg;
    return *this;
  }
  Type & set__armed(
    const bool & _arg)
  {
    this->armed = _arg;
    return *this;
  }
  Type & set__offboard(
    const bool & _arg)
  {
    this->offboard = _arg;
    return *this;
  }
  Type & set__status(
    const as2_msgs::msg::PlatformStatus_<ContainerAllocator> & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__current_control_mode(
    const as2_msgs::msg::ControlMode_<ContainerAllocator> & _arg)
  {
    this->current_control_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::PlatformInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::PlatformInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::PlatformInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::PlatformInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__PlatformInfo
    std::shared_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__PlatformInfo
    std::shared_ptr<as2_msgs::msg::PlatformInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PlatformInfo_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->connected != other.connected) {
      return false;
    }
    if (this->armed != other.armed) {
      return false;
    }
    if (this->offboard != other.offboard) {
      return false;
    }
    if (this->status != other.status) {
      return false;
    }
    if (this->current_control_mode != other.current_control_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const PlatformInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PlatformInfo_

// alias to use template instance with default allocator
using PlatformInfo =
  as2_msgs::msg::PlatformInfo_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__PLATFORM_INFO__STRUCT_HPP_
