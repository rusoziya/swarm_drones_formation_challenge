// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/TrajectoryPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'position'
// Member 'twist'
// Member 'acceleration'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__TrajectoryPoint __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__TrajectoryPoint __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TrajectoryPoint_
{
  using Type = TrajectoryPoint_<ContainerAllocator>;

  explicit TrajectoryPoint_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init),
    twist(_init),
    acceleration(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->yaw_angle = 0.0f;
    }
  }

  explicit TrajectoryPoint_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init),
    twist(_alloc, _init),
    acceleration(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->yaw_angle = 0.0f;
    }
  }

  // field types and members
  using _position_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _position_type position;
  using _twist_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _twist_type twist;
  using _acceleration_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _acceleration_type acceleration;
  using _yaw_angle_type =
    float;
  _yaw_angle_type yaw_angle;

  // setters for named parameter idiom
  Type & set__position(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__twist(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->twist = _arg;
    return *this;
  }
  Type & set__acceleration(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->acceleration = _arg;
    return *this;
  }
  Type & set__yaw_angle(
    const float & _arg)
  {
    this->yaw_angle = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::TrajectoryPoint_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::TrajectoryPoint_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::TrajectoryPoint_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::TrajectoryPoint_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__TrajectoryPoint
    std::shared_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__TrajectoryPoint
    std::shared_ptr<as2_msgs::msg::TrajectoryPoint_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TrajectoryPoint_ & other) const
  {
    if (this->position != other.position) {
      return false;
    }
    if (this->twist != other.twist) {
      return false;
    }
    if (this->acceleration != other.acceleration) {
      return false;
    }
    if (this->yaw_angle != other.yaw_angle) {
      return false;
    }
    return true;
  }
  bool operator!=(const TrajectoryPoint_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TrajectoryPoint_

// alias to use template instance with default allocator
using TrajectoryPoint =
  as2_msgs::msg::TrajectoryPoint_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__TRAJECTORY_POINT__STRUCT_HPP_
