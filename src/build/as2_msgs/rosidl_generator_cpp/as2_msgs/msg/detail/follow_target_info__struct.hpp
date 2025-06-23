// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/FollowTargetInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__STRUCT_HPP_

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
# define DEPRECATED__as2_msgs__msg__FollowTargetInfo __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__FollowTargetInfo __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct FollowTargetInfo_
{
  using Type = FollowTargetInfo_<ContainerAllocator>;

  explicit FollowTargetInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->follow_status = 0;
      this->follow_mode = 0;
    }
  }

  explicit FollowTargetInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->follow_status = 0;
      this->follow_mode = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _follow_status_type =
    int8_t;
  _follow_status_type follow_status;
  using _follow_mode_type =
    int8_t;
  _follow_mode_type follow_mode;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__follow_status(
    const int8_t & _arg)
  {
    this->follow_status = _arg;
    return *this;
  }
  Type & set__follow_mode(
    const int8_t & _arg)
  {
    this->follow_mode = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t WAITING =
    0;
  static constexpr int8_t RUNNING =
    1;
  static constexpr int8_t END =
    2;
  static constexpr int8_t UNSET =
    0;
  static constexpr int8_t PICKUP =
    1;
  static constexpr int8_t UNPICK =
    2;
  static constexpr int8_t DYNAMIC_LAND =
    3;
  static constexpr int8_t DYNAMIC_FOLLOWER =
    4;

  // pointer types
  using RawPtr =
    as2_msgs::msg::FollowTargetInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::FollowTargetInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::FollowTargetInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::FollowTargetInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__FollowTargetInfo
    std::shared_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__FollowTargetInfo
    std::shared_ptr<as2_msgs::msg::FollowTargetInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const FollowTargetInfo_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->follow_status != other.follow_status) {
      return false;
    }
    if (this->follow_mode != other.follow_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const FollowTargetInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct FollowTargetInfo_

// alias to use template instance with default allocator
using FollowTargetInfo =
  as2_msgs::msg::FollowTargetInfo_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::WAITING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::RUNNING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::END;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::UNSET;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::PICKUP;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::UNPICK;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::DYNAMIC_LAND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t FollowTargetInfo_<ContainerAllocator>::DYNAMIC_FOLLOWER;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__FOLLOW_TARGET_INFO__STRUCT_HPP_
