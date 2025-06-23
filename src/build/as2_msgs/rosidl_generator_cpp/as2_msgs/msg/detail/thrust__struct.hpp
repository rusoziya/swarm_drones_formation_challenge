// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/Thrust.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__THRUST__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__THRUST__STRUCT_HPP_

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
# define DEPRECATED__as2_msgs__msg__Thrust __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__Thrust __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Thrust_
{
  using Type = Thrust_<ContainerAllocator>;

  explicit Thrust_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->thrust = 0.0f;
      this->thrust_normalized = 0.0f;
    }
  }

  explicit Thrust_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->thrust = 0.0f;
      this->thrust_normalized = 0.0f;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _thrust_type =
    float;
  _thrust_type thrust;
  using _thrust_normalized_type =
    float;
  _thrust_normalized_type thrust_normalized;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__thrust(
    const float & _arg)
  {
    this->thrust = _arg;
    return *this;
  }
  Type & set__thrust_normalized(
    const float & _arg)
  {
    this->thrust_normalized = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::Thrust_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::Thrust_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::Thrust_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::Thrust_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::Thrust_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::Thrust_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::Thrust_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::Thrust_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::Thrust_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::Thrust_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__Thrust
    std::shared_ptr<as2_msgs::msg::Thrust_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__Thrust
    std::shared_ptr<as2_msgs::msg::Thrust_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Thrust_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->thrust != other.thrust) {
      return false;
    }
    if (this->thrust_normalized != other.thrust_normalized) {
      return false;
    }
    return true;
  }
  bool operator!=(const Thrust_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Thrust_

// alias to use template instance with default allocator
using Thrust =
  as2_msgs::msg::Thrust_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__THRUST__STRUCT_HPP_
