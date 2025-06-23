// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/Speed.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__SPEED__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__SPEED__STRUCT_HPP_

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

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__Speed __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__Speed __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Speed_
{
  using Type = Speed_<ContainerAllocator>;

  explicit Speed_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->speed = 0.0f;
    }
  }

  explicit Speed_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->speed = 0.0f;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _speed_type =
    float;
  _speed_type speed;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__speed(
    const float & _arg)
  {
    this->speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::Speed_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::Speed_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::Speed_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::Speed_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::Speed_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::Speed_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::Speed_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::Speed_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::Speed_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::Speed_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__Speed
    std::shared_ptr<as2_msgs::msg::Speed_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__Speed
    std::shared_ptr<as2_msgs::msg::Speed_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Speed_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->speed != other.speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const Speed_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Speed_

// alias to use template instance with default allocator
using Speed =
  as2_msgs::msg::Speed_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__SPEED__STRUCT_HPP_
