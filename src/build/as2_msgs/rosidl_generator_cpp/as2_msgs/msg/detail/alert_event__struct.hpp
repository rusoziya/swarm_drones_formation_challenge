// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/AlertEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ALERT_EVENT__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__ALERT_EVENT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__AlertEvent __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__AlertEvent __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct AlertEvent_
{
  using Type = AlertEvent_<ContainerAllocator>;

  explicit AlertEvent_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->alert = 0;
      this->description = "";
    }
  }

  explicit AlertEvent_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : description(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->alert = 0;
      this->description = "";
    }
  }

  // field types and members
  using _alert_type =
    int8_t;
  _alert_type alert;
  using _description_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _description_type description;

  // setters for named parameter idiom
  Type & set__alert(
    const int8_t & _arg)
  {
    this->alert = _arg;
    return *this;
  }
  Type & set__description(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->description = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t KILL_SWITCH =
    -1;
  static constexpr int8_t EMERGENCY_HOVER =
    -2;
  static constexpr int8_t EMERGENCY_LAND =
    -3;
  static constexpr int8_t INFO_ALERT =
    0;
  static constexpr int8_t FORCE_HOVER =
    1;
  static constexpr int8_t FORCE_LAND =
    2;

  // pointer types
  using RawPtr =
    as2_msgs::msg::AlertEvent_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::AlertEvent_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::AlertEvent_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::AlertEvent_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__AlertEvent
    std::shared_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__AlertEvent
    std::shared_ptr<as2_msgs::msg::AlertEvent_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AlertEvent_ & other) const
  {
    if (this->alert != other.alert) {
      return false;
    }
    if (this->description != other.description) {
      return false;
    }
    return true;
  }
  bool operator!=(const AlertEvent_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AlertEvent_

// alias to use template instance with default allocator
using AlertEvent =
  as2_msgs::msg::AlertEvent_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t AlertEvent_<ContainerAllocator>::KILL_SWITCH;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t AlertEvent_<ContainerAllocator>::EMERGENCY_HOVER;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t AlertEvent_<ContainerAllocator>::EMERGENCY_LAND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t AlertEvent_<ContainerAllocator>::INFO_ALERT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t AlertEvent_<ContainerAllocator>::FORCE_HOVER;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t AlertEvent_<ContainerAllocator>::FORCE_LAND;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__ALERT_EVENT__STRUCT_HPP_
