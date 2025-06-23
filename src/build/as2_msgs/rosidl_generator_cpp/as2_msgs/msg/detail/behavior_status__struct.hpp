// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/BehaviorStatus.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__BehaviorStatus __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__BehaviorStatus __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct BehaviorStatus_
{
  using Type = BehaviorStatus_<ContainerAllocator>;

  explicit BehaviorStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  explicit BehaviorStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    uint8_t;
  _status_type status;

  // setters for named parameter idiom
  Type & set__status(
    const uint8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t IDLE =
    0u;
  static constexpr uint8_t RUNNING =
    1u;
  static constexpr uint8_t PAUSED =
    2u;

  // pointer types
  using RawPtr =
    as2_msgs::msg::BehaviorStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::BehaviorStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::BehaviorStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::BehaviorStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__BehaviorStatus
    std::shared_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__BehaviorStatus
    std::shared_ptr<as2_msgs::msg::BehaviorStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const BehaviorStatus_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    return true;
  }
  bool operator!=(const BehaviorStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct BehaviorStatus_

// alias to use template instance with default allocator
using BehaviorStatus =
  as2_msgs::msg::BehaviorStatus_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t BehaviorStatus_<ContainerAllocator>::IDLE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t BehaviorStatus_<ContainerAllocator>::RUNNING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t BehaviorStatus_<ContainerAllocator>::PAUSED;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__BEHAVIOR_STATUS__STRUCT_HPP_
