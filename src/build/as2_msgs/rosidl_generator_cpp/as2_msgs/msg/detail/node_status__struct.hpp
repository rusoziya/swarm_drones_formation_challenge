// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/NodeStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__NODE_STATUS__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__NODE_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__NodeStatus __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__NodeStatus __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct NodeStatus_
{
  using Type = NodeStatus_<ContainerAllocator>;

  explicit NodeStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  explicit NodeStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  // field types and members
  using _status_type =
    int8_t;
  _status_type status;

  // setters for named parameter idiom
  Type & set__status(
    const int8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t UNCONFIGURED =
    0;
  static constexpr int8_t INACTIVE =
    1;
  static constexpr int8_t ACTIVE =
    2;
  static constexpr int8_t FINALIZED =
    3;

  // pointer types
  using RawPtr =
    as2_msgs::msg::NodeStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::NodeStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::NodeStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::NodeStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__NodeStatus
    std::shared_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__NodeStatus
    std::shared_ptr<as2_msgs::msg::NodeStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NodeStatus_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    return true;
  }
  bool operator!=(const NodeStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NodeStatus_

// alias to use template instance with default allocator
using NodeStatus =
  as2_msgs::msg::NodeStatus_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t NodeStatus_<ContainerAllocator>::UNCONFIGURED;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t NodeStatus_<ContainerAllocator>::INACTIVE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t NodeStatus_<ContainerAllocator>::ACTIVE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t NodeStatus_<ContainerAllocator>::FINALIZED;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__NODE_STATUS__STRUCT_HPP_
