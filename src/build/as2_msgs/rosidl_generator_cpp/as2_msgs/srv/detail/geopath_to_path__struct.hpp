// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/GeopathToPath.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__GEOPATH_TO_PATH__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__GEOPATH_TO_PATH__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'geo_path'
#include "geographic_msgs/msg/detail/geo_path__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__GeopathToPath_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__GeopathToPath_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GeopathToPath_Request_
{
  using Type = GeopathToPath_Request_<ContainerAllocator>;

  explicit GeopathToPath_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : geo_path(_init)
  {
    (void)_init;
  }

  explicit GeopathToPath_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : geo_path(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _geo_path_type =
    geographic_msgs::msg::GeoPath_<ContainerAllocator>;
  _geo_path_type geo_path;

  // setters for named parameter idiom
  Type & set__geo_path(
    const geographic_msgs::msg::GeoPath_<ContainerAllocator> & _arg)
  {
    this->geo_path = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__GeopathToPath_Request
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__GeopathToPath_Request
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GeopathToPath_Request_ & other) const
  {
    if (this->geo_path != other.geo_path) {
      return false;
    }
    return true;
  }
  bool operator!=(const GeopathToPath_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GeopathToPath_Request_

// alias to use template instance with default allocator
using GeopathToPath_Request =
  as2_msgs::srv::GeopathToPath_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


// Include directives for member types
// Member 'path'
#include "nav_msgs/msg/detail/path__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__GeopathToPath_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__GeopathToPath_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GeopathToPath_Response_
{
  using Type = GeopathToPath_Response_<ContainerAllocator>;

  explicit GeopathToPath_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : path(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit GeopathToPath_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : path(_alloc, _init)
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
  using _path_type =
    nav_msgs::msg::Path_<ContainerAllocator>;
  _path_type path;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__path(
    const nav_msgs::msg::Path_<ContainerAllocator> & _arg)
  {
    this->path = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__GeopathToPath_Response
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__GeopathToPath_Response
    std::shared_ptr<as2_msgs::srv::GeopathToPath_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GeopathToPath_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->path != other.path) {
      return false;
    }
    return true;
  }
  bool operator!=(const GeopathToPath_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GeopathToPath_Response_

// alias to use template instance with default allocator
using GeopathToPath_Response =
  as2_msgs::srv::GeopathToPath_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct GeopathToPath
{
  using Request = as2_msgs::srv::GeopathToPath_Request;
  using Response = as2_msgs::srv::GeopathToPath_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__GEOPATH_TO_PATH__STRUCT_HPP_
