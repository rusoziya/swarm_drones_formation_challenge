// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/GetGeozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__GET_GEOZONE__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__GET_GEOZONE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__GetGeozone_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__GetGeozone_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetGeozone_Request_
{
  using Type = GetGeozone_Request_<ContainerAllocator>;

  explicit GetGeozone_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit GetGeozone_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::GetGeozone_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::GetGeozone_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GetGeozone_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GetGeozone_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__GetGeozone_Request
    std::shared_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__GetGeozone_Request
    std::shared_ptr<as2_msgs::srv::GetGeozone_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetGeozone_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetGeozone_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetGeozone_Request_

// alias to use template instance with default allocator
using GetGeozone_Request =
  as2_msgs::srv::GetGeozone_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


// Include directives for member types
// Member 'geozone_list'
#include "as2_msgs/msg/detail/geozone__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__GetGeozone_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__GetGeozone_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetGeozone_Response_
{
  using Type = GetGeozone_Response_<ContainerAllocator>;

  explicit GetGeozone_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit GetGeozone_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
  using _geozone_list_type =
    std::vector<as2_msgs::msg::Geozone_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<as2_msgs::msg::Geozone_<ContainerAllocator>>>;
  _geozone_list_type geozone_list;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__geozone_list(
    const std::vector<as2_msgs::msg::Geozone_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<as2_msgs::msg::Geozone_<ContainerAllocator>>> & _arg)
  {
    this->geozone_list = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::GetGeozone_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::GetGeozone_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GetGeozone_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::GetGeozone_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__GetGeozone_Response
    std::shared_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__GetGeozone_Response
    std::shared_ptr<as2_msgs::srv::GetGeozone_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetGeozone_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->geozone_list != other.geozone_list) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetGeozone_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetGeozone_Response_

// alias to use template instance with default allocator
using GetGeozone_Response =
  as2_msgs::srv::GetGeozone_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct GetGeozone
{
  using Request = as2_msgs::srv::GetGeozone_Request;
  using Response = as2_msgs::srv::GetGeozone_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__GET_GEOZONE__STRUCT_HPP_
