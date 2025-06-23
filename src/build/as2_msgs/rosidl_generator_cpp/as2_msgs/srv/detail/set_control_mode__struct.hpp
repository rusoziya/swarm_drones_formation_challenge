// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/SetControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'control_mode'
#include "as2_msgs/msg/detail/control_mode__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__SetControlMode_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__SetControlMode_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetControlMode_Request_
{
  using Type = SetControlMode_Request_<ContainerAllocator>;

  explicit SetControlMode_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : control_mode(_init)
  {
    (void)_init;
  }

  explicit SetControlMode_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : control_mode(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _control_mode_type =
    as2_msgs::msg::ControlMode_<ContainerAllocator>;
  _control_mode_type control_mode;

  // setters for named parameter idiom
  Type & set__control_mode(
    const as2_msgs::msg::ControlMode_<ContainerAllocator> & _arg)
  {
    this->control_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::SetControlMode_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::SetControlMode_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetControlMode_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetControlMode_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__SetControlMode_Request
    std::shared_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__SetControlMode_Request
    std::shared_ptr<as2_msgs::srv::SetControlMode_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetControlMode_Request_ & other) const
  {
    if (this->control_mode != other.control_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetControlMode_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetControlMode_Request_

// alias to use template instance with default allocator
using SetControlMode_Request =
  as2_msgs::srv::SetControlMode_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__SetControlMode_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__SetControlMode_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetControlMode_Response_
{
  using Type = SetControlMode_Response_<ContainerAllocator>;

  explicit SetControlMode_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetControlMode_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::srv::SetControlMode_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::SetControlMode_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetControlMode_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetControlMode_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__SetControlMode_Response
    std::shared_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__SetControlMode_Response
    std::shared_ptr<as2_msgs::srv::SetControlMode_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetControlMode_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetControlMode_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetControlMode_Response_

// alias to use template instance with default allocator
using SetControlMode_Response =
  as2_msgs::srv::SetControlMode_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct SetControlMode
{
  using Request = as2_msgs::srv::SetControlMode_Request;
  using Response = as2_msgs::srv::SetControlMode_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__STRUCT_HPP_
