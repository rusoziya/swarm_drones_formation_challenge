// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/TrajGenInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__STRUCT_HPP_

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
// Member 'node_status'
#include "as2_msgs/msg/detail/node_status__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__TrajGenInfo __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__TrajGenInfo __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TrajGenInfo_
{
  using Type = TrajGenInfo_<ContainerAllocator>;

  explicit TrajGenInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    node_status(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->active_status = 0;
    }
  }

  explicit TrajGenInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    node_status(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->active_status = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _node_status_type =
    as2_msgs::msg::NodeStatus_<ContainerAllocator>;
  _node_status_type node_status;
  using _active_status_type =
    uint8_t;
  _active_status_type active_status;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__node_status(
    const as2_msgs::msg::NodeStatus_<ContainerAllocator> & _arg)
  {
    this->node_status = _arg;
    return *this;
  }
  Type & set__active_status(
    const uint8_t & _arg)
  {
    this->active_status = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t WAITING =
    0u;
  static constexpr uint8_t EVALUATING =
    1u;
  static constexpr uint8_t STOPPED =
    2u;

  // pointer types
  using RawPtr =
    as2_msgs::msg::TrajGenInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::TrajGenInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::TrajGenInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::TrajGenInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__TrajGenInfo
    std::shared_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__TrajGenInfo
    std::shared_ptr<as2_msgs::msg::TrajGenInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TrajGenInfo_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->node_status != other.node_status) {
      return false;
    }
    if (this->active_status != other.active_status) {
      return false;
    }
    return true;
  }
  bool operator!=(const TrajGenInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TrajGenInfo_

// alias to use template instance with default allocator
using TrajGenInfo =
  as2_msgs::msg::TrajGenInfo_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t TrajGenInfo_<ContainerAllocator>::WAITING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t TrajGenInfo_<ContainerAllocator>::EVALUATING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t TrajGenInfo_<ContainerAllocator>::STOPPED;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__TRAJ_GEN_INFO__STRUCT_HPP_
