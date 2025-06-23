// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/PoseWithID.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__PoseWithID __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__PoseWithID __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PoseWithID_
{
  using Type = PoseWithID_<ContainerAllocator>;

  explicit PoseWithID_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = "";
    }
  }

  explicit PoseWithID_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : id(_alloc),
    pose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = "";
    }
  }

  // field types and members
  using _id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _id_type id;
  using _pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pose_type pose;

  // setters for named parameter idiom
  Type & set__id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::PoseWithID_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::PoseWithID_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::PoseWithID_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::PoseWithID_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__PoseWithID
    std::shared_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__PoseWithID
    std::shared_ptr<as2_msgs::msg::PoseWithID_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PoseWithID_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->pose != other.pose) {
      return false;
    }
    return true;
  }
  bool operator!=(const PoseWithID_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PoseWithID_

// alias to use template instance with default allocator
using PoseWithID =
  as2_msgs::msg::PoseWithID_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__POSE_WITH_ID__STRUCT_HPP_
