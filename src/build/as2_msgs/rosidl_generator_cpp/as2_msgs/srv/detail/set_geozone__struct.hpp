// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/SetGeozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_GEOZONE__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__SET_GEOZONE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'geozone'
#include "as2_msgs/msg/detail/geozone__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__SetGeozone_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__SetGeozone_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetGeozone_Request_
{
  using Type = SetGeozone_Request_<ContainerAllocator>;

  explicit SetGeozone_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : geozone(_init)
  {
    (void)_init;
  }

  explicit SetGeozone_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : geozone(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _geozone_type =
    as2_msgs::msg::Geozone_<ContainerAllocator>;
  _geozone_type geozone;

  // setters for named parameter idiom
  Type & set__geozone(
    const as2_msgs::msg::Geozone_<ContainerAllocator> & _arg)
  {
    this->geozone = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::SetGeozone_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::SetGeozone_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetGeozone_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetGeozone_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__SetGeozone_Request
    std::shared_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__SetGeozone_Request
    std::shared_ptr<as2_msgs::srv::SetGeozone_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetGeozone_Request_ & other) const
  {
    if (this->geozone != other.geozone) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetGeozone_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetGeozone_Request_

// alias to use template instance with default allocator
using SetGeozone_Request =
  as2_msgs::srv::SetGeozone_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__SetGeozone_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__SetGeozone_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SetGeozone_Response_
{
  using Type = SetGeozone_Response_<ContainerAllocator>;

  explicit SetGeozone_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit SetGeozone_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::srv::SetGeozone_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::SetGeozone_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetGeozone_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::SetGeozone_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__SetGeozone_Response
    std::shared_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__SetGeozone_Response
    std::shared_ptr<as2_msgs::srv::SetGeozone_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SetGeozone_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const SetGeozone_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SetGeozone_Response_

// alias to use template instance with default allocator
using SetGeozone_Response =
  as2_msgs::srv::SetGeozone_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct SetGeozone
{
  using Request = as2_msgs::srv::SetGeozone_Request;
  using Response = as2_msgs::srv::SetGeozone_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__SET_GEOZONE__STRUCT_HPP_
