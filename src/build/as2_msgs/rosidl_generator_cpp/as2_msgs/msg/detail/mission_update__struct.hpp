// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/MissionUpdate.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__MissionUpdate __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__MissionUpdate __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MissionUpdate_
{
  using Type = MissionUpdate_<ContainerAllocator>;

  explicit MissionUpdate_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->drone_id = "";
      this->mission_id = 0l;
      this->item_id = 0l;
      this->action = 0;
      this->mission = "";
    }
  }

  explicit MissionUpdate_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : drone_id(_alloc),
    mission(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->drone_id = "";
      this->mission_id = 0l;
      this->item_id = 0l;
      this->action = 0;
      this->mission = "";
    }
  }

  // field types and members
  using _drone_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _drone_id_type drone_id;
  using _mission_id_type =
    int32_t;
  _mission_id_type mission_id;
  using _item_id_type =
    int32_t;
  _item_id_type item_id;
  using _action_type =
    uint8_t;
  _action_type action;
  using _mission_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _mission_type mission;

  // setters for named parameter idiom
  Type & set__drone_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->drone_id = _arg;
    return *this;
  }
  Type & set__mission_id(
    const int32_t & _arg)
  {
    this->mission_id = _arg;
    return *this;
  }
  Type & set__item_id(
    const int32_t & _arg)
  {
    this->item_id = _arg;
    return *this;
  }
  Type & set__action(
    const uint8_t & _arg)
  {
    this->action = _arg;
    return *this;
  }
  Type & set__mission(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->mission = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t EXECUTE =
    0u;
  static constexpr uint8_t LOAD =
    1u;
  static constexpr uint8_t START =
    2u;
  static constexpr uint8_t PAUSE =
    3u;
  static constexpr uint8_t RESUME =
    4u;
  static constexpr uint8_t STOP =
    5u;
  static constexpr uint8_t JUMP_TO =
    6u;
  static constexpr uint8_t REPEAT =
    7u;
  static constexpr uint8_t INSERT =
    8u;
  static constexpr uint8_t MODIFY =
    9u;
  static constexpr uint8_t REMOVE =
    10u;
  static constexpr uint8_t RESET =
    11u;
  static constexpr uint8_t ABORT =
    12u;

  // pointer types
  using RawPtr =
    as2_msgs::msg::MissionUpdate_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::MissionUpdate_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::MissionUpdate_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::MissionUpdate_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__MissionUpdate
    std::shared_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__MissionUpdate
    std::shared_ptr<as2_msgs::msg::MissionUpdate_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MissionUpdate_ & other) const
  {
    if (this->drone_id != other.drone_id) {
      return false;
    }
    if (this->mission_id != other.mission_id) {
      return false;
    }
    if (this->item_id != other.item_id) {
      return false;
    }
    if (this->action != other.action) {
      return false;
    }
    if (this->mission != other.mission) {
      return false;
    }
    return true;
  }
  bool operator!=(const MissionUpdate_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MissionUpdate_

// alias to use template instance with default allocator
using MissionUpdate =
  as2_msgs::msg::MissionUpdate_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::EXECUTE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::LOAD;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::START;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::PAUSE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::RESUME;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::STOP;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::JUMP_TO;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::REPEAT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::INSERT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::MODIFY;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::REMOVE;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::RESET;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t MissionUpdate_<ContainerAllocator>::ABORT;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__STRUCT_HPP_
