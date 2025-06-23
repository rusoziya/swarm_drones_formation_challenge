// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/ControllerInfo.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__STRUCT_HPP_

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
// Member 'input_control_mode'
// Member 'output_control_mode'
#include "as2_msgs/msg/detail/control_mode__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__ControllerInfo __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__ControllerInfo __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ControllerInfo_
{
  using Type = ControllerInfo_<ContainerAllocator>;

  explicit ControllerInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    input_control_mode(_init),
    output_control_mode(_init)
  {
    (void)_init;
  }

  explicit ControllerInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    input_control_mode(_alloc, _init),
    output_control_mode(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _input_control_mode_type =
    as2_msgs::msg::ControlMode_<ContainerAllocator>;
  _input_control_mode_type input_control_mode;
  using _output_control_mode_type =
    as2_msgs::msg::ControlMode_<ContainerAllocator>;
  _output_control_mode_type output_control_mode;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__input_control_mode(
    const as2_msgs::msg::ControlMode_<ContainerAllocator> & _arg)
  {
    this->input_control_mode = _arg;
    return *this;
  }
  Type & set__output_control_mode(
    const as2_msgs::msg::ControlMode_<ContainerAllocator> & _arg)
  {
    this->output_control_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::ControllerInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::ControllerInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::ControllerInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::ControllerInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__ControllerInfo
    std::shared_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__ControllerInfo
    std::shared_ptr<as2_msgs::msg::ControllerInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ControllerInfo_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->input_control_mode != other.input_control_mode) {
      return false;
    }
    if (this->output_control_mode != other.output_control_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const ControllerInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ControllerInfo_

// alias to use template instance with default allocator
using ControllerInfo =
  as2_msgs::msg::ControllerInfo_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__CONTROLLER_INFO__STRUCT_HPP_
