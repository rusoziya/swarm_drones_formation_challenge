// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/SetPlatformStateMachineEvent.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'event'
#include "as2_msgs/msg/detail/platform_state_machine_event__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetPlatformStateMachineEvent_Request_
{
  using Type = SetPlatformStateMachineEvent_Request_<ContainerAllocator>;

  explicit SetPlatformStateMachineEvent_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : event(_init)
  {
    (void)_init;
  }

  explicit SetPlatformStateMachineEvent_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : event(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _event_type =
    as2_msgs::msg::PlatformStateMachineEvent_<ContainerAllocator>;
  _event_type event;

  // setters for named parameter idiom
  Type & set__event(
    const as2_msgs::msg::PlatformStateMachineEvent_<ContainerAllocator> & _arg)
  {
    this->event = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Request
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Request
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetPlatformStateMachineEvent_Request_ & other) const
  {
    if (this->event != other.event) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetPlatformStateMachineEvent_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetPlatformStateMachineEvent_Request_

// alias to use template instance with default allocator
using SetPlatformStateMachineEvent_Request =
  as2_msgs::srv::SetPlatformStateMachineEvent_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


// Include directives for member types
// Member 'current_state'
#include "as2_msgs/msg/detail/platform_status__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetPlatformStateMachineEvent_Response_
{
  using Type = SetPlatformStateMachineEvent_Response_<ContainerAllocator>;

  explicit SetPlatformStateMachineEvent_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : current_state(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetPlatformStateMachineEvent_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : current_state(_alloc, _init)
  {
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
  using _current_state_type =
    as2_msgs::msg::PlatformStatus_<ContainerAllocator>;
  _current_state_type current_state;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__current_state(
    const as2_msgs::msg::PlatformStatus_<ContainerAllocator> & _arg)
  {
    this->current_state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Response
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__SetPlatformStateMachineEvent_Response
    std::shared_ptr<as2_msgs::srv::SetPlatformStateMachineEvent_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetPlatformStateMachineEvent_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->current_state != other.current_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetPlatformStateMachineEvent_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetPlatformStateMachineEvent_Response_

// alias to use template instance with default allocator
using SetPlatformStateMachineEvent_Response =
  as2_msgs::srv::SetPlatformStateMachineEvent_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct SetPlatformStateMachineEvent
{
  using Request = as2_msgs::srv::SetPlatformStateMachineEvent_Request;
  using Response = as2_msgs::srv::SetPlatformStateMachineEvent_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_PLATFORM_STATE_MACHINE_EVENT__STRUCT_HPP_
