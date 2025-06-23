// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/ListControlModes.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__ListControlModes_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__ListControlModes_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ListControlModes_Request_
{
  using Type = ListControlModes_Request_<ContainerAllocator>;

  explicit ListControlModes_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit ListControlModes_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::srv::ListControlModes_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::ListControlModes_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::ListControlModes_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::ListControlModes_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__ListControlModes_Request
    std::shared_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__ListControlModes_Request
    std::shared_ptr<as2_msgs::srv::ListControlModes_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ListControlModes_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const ListControlModes_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ListControlModes_Request_

// alias to use template instance with default allocator
using ListControlModes_Request =
  as2_msgs::srv::ListControlModes_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__ListControlModes_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__ListControlModes_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ListControlModes_Response_
{
  using Type = ListControlModes_Response_<ContainerAllocator>;

  explicit ListControlModes_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->source = "";
    }
  }

  explicit ListControlModes_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : source(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->source = "";
    }
  }

  // field types and members
  using _source_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _source_type source;
  using _control_modes_type =
    std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>>;
  _control_modes_type control_modes;

  // setters for named parameter idiom
  Type & set__source(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->source = _arg;
    return *this;
  }
  Type & set__control_modes(
    const std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>> & _arg)
  {
    this->control_modes = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::ListControlModes_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::ListControlModes_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::ListControlModes_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::ListControlModes_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__ListControlModes_Response
    std::shared_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__ListControlModes_Response
    std::shared_ptr<as2_msgs::srv::ListControlModes_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ListControlModes_Response_ & other) const
  {
    if (this->source != other.source) {
      return false;
    }
    if (this->control_modes != other.control_modes) {
      return false;
    }
    return true;
  }
  bool operator!=(const ListControlModes_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ListControlModes_Response_

// alias to use template instance with default allocator
using ListControlModes_Response =
  as2_msgs::srv::ListControlModes_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct ListControlModes
{
  using Request = as2_msgs::srv::ListControlModes_Request;
  using Response = as2_msgs::srv::ListControlModes_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__LIST_CONTROL_MODES__STRUCT_HPP_
