// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/ControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROL_MODE__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__CONTROL_MODE__STRUCT_HPP_

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
# define DEPRECATED__as2_msgs__msg__ControlMode __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__ControlMode __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ControlMode_
{
  using Type = ControlMode_<ContainerAllocator>;

  explicit ControlMode_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->yaw_mode = 0;
      this->control_mode = 0;
      this->reference_frame = 0;
    }
  }

  explicit ControlMode_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->yaw_mode = 0;
      this->control_mode = 0;
      this->reference_frame = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _yaw_mode_type =
    int8_t;
  _yaw_mode_type yaw_mode;
  using _control_mode_type =
    int8_t;
  _control_mode_type control_mode;
  using _reference_frame_type =
    int8_t;
  _reference_frame_type reference_frame;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__yaw_mode(
    const int8_t & _arg)
  {
    this->yaw_mode = _arg;
    return *this;
  }
  Type & set__control_mode(
    const int8_t & _arg)
  {
    this->control_mode = _arg;
    return *this;
  }
  Type & set__reference_frame(
    const int8_t & _arg)
  {
    this->reference_frame = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t NONE =
    0;
  static constexpr int8_t YAW_ANGLE =
    1;
  static constexpr int8_t YAW_SPEED =
    2;
  static constexpr int8_t UNSET =
    0;
  static constexpr int8_t HOVER =
    1;
  static constexpr int8_t POSITION =
    2;
  static constexpr int8_t SPEED =
    3;
  static constexpr int8_t SPEED_IN_A_PLANE =
    4;
  static constexpr int8_t ATTITUDE =
    5;
  static constexpr int8_t ACRO =
    6;
  static constexpr int8_t TRAJECTORY =
    7;
  static constexpr int8_t UNDEFINED_FRAME =
    0;
  static constexpr int8_t LOCAL_ENU_FRAME =
    1;
  static constexpr int8_t BODY_FLU_FRAME =
    2;
  static constexpr int8_t GLOBAL_LAT_LONG_ASML =
    3;

  // pointer types
  using RawPtr =
    as2_msgs::msg::ControlMode_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::ControlMode_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::ControlMode_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::ControlMode_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__ControlMode
    std::shared_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__ControlMode
    std::shared_ptr<as2_msgs::msg::ControlMode_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ControlMode_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->yaw_mode != other.yaw_mode) {
      return false;
    }
    if (this->control_mode != other.control_mode) {
      return false;
    }
    if (this->reference_frame != other.reference_frame) {
      return false;
    }
    return true;
  }
  bool operator!=(const ControlMode_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ControlMode_

// alias to use template instance with default allocator
using ControlMode =
  as2_msgs::msg::ControlMode_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::NONE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::YAW_ANGLE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::YAW_SPEED;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::UNSET;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::HOVER;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::POSITION;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::SPEED;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::SPEED_IN_A_PLANE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::ATTITUDE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::ACRO;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::TRAJECTORY;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::UNDEFINED_FRAME;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::LOCAL_ENU_FRAME;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::BODY_FLU_FRAME;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t ControlMode_<ContainerAllocator>::GLOBAL_LAT_LONG_ASML;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__CONTROL_MODE__STRUCT_HPP_
