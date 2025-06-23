// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/DynamicFollower.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__DYNAMIC_FOLLOWER__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__DYNAMIC_FOLLOWER__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'speed_limit'
#include "geometry_msgs/msg/detail/twist__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__DynamicFollower_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__DynamicFollower_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct DynamicFollower_Request_
{
  using Type = DynamicFollower_Request_<ContainerAllocator>;

  explicit DynamicFollower_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : speed_limit(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->enable = false;
    }
  }

  explicit DynamicFollower_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : speed_limit(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->enable = false;
    }
  }

  // field types and members
  using _enable_type =
    bool;
  _enable_type enable;
  using _speed_limit_type =
    geometry_msgs::msg::Twist_<ContainerAllocator>;
  _speed_limit_type speed_limit;

  // setters for named parameter idiom
  Type & set__enable(
    const bool & _arg)
  {
    this->enable = _arg;
    return *this;
  }
  Type & set__speed_limit(
    const geometry_msgs::msg::Twist_<ContainerAllocator> & _arg)
  {
    this->speed_limit = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__DynamicFollower_Request
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__DynamicFollower_Request
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DynamicFollower_Request_ & other) const
  {
    if (this->enable != other.enable) {
      return false;
    }
    if (this->speed_limit != other.speed_limit) {
      return false;
    }
    return true;
  }
  bool operator!=(const DynamicFollower_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DynamicFollower_Request_

// alias to use template instance with default allocator
using DynamicFollower_Request =
  as2_msgs::srv::DynamicFollower_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__DynamicFollower_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__DynamicFollower_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct DynamicFollower_Response_
{
  using Type = DynamicFollower_Response_<ContainerAllocator>;

  explicit DynamicFollower_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit DynamicFollower_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__DynamicFollower_Response
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__DynamicFollower_Response
    std::shared_ptr<as2_msgs::srv::DynamicFollower_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DynamicFollower_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const DynamicFollower_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DynamicFollower_Response_

// alias to use template instance with default allocator
using DynamicFollower_Response =
  as2_msgs::srv::DynamicFollower_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct DynamicFollower
{
  using Request = as2_msgs::srv::DynamicFollower_Request;
  using Response = as2_msgs::srv::DynamicFollower_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__DYNAMIC_FOLLOWER__STRUCT_HPP_
