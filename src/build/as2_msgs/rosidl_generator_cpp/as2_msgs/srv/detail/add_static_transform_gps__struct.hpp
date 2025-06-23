// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:srv/AddStaticTransformGps.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__STRUCT_HPP_
#define AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'gps_position'
#include "sensor_msgs/msg/detail/nav_sat_fix__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct AddStaticTransformGps_Request_
{
  using Type = AddStaticTransformGps_Request_<ContainerAllocator>;

  explicit AddStaticTransformGps_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : gps_position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->child_frame_id = "";
      this->azimuth = 0.0f;
      this->elevation = 0.0f;
      this->bank = 0.0f;
    }
  }

  explicit AddStaticTransformGps_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : frame_id(_alloc),
    child_frame_id(_alloc),
    gps_position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->child_frame_id = "";
      this->azimuth = 0.0f;
      this->elevation = 0.0f;
      this->bank = 0.0f;
    }
  }

  // field types and members
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _child_frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _child_frame_id_type child_frame_id;
  using _gps_position_type =
    sensor_msgs::msg::NavSatFix_<ContainerAllocator>;
  _gps_position_type gps_position;
  using _azimuth_type =
    float;
  _azimuth_type azimuth;
  using _elevation_type =
    float;
  _elevation_type elevation;
  using _bank_type =
    float;
  _bank_type bank;

  // setters for named parameter idiom
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__child_frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->child_frame_id = _arg;
    return *this;
  }
  Type & set__gps_position(
    const sensor_msgs::msg::NavSatFix_<ContainerAllocator> & _arg)
  {
    this->gps_position = _arg;
    return *this;
  }
  Type & set__azimuth(
    const float & _arg)
  {
    this->azimuth = _arg;
    return *this;
  }
  Type & set__elevation(
    const float & _arg)
  {
    this->elevation = _arg;
    return *this;
  }
  Type & set__bank(
    const float & _arg)
  {
    this->bank = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Request
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Request
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AddStaticTransformGps_Request_ & other) const
  {
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->child_frame_id != other.child_frame_id) {
      return false;
    }
    if (this->gps_position != other.gps_position) {
      return false;
    }
    if (this->azimuth != other.azimuth) {
      return false;
    }
    if (this->elevation != other.elevation) {
      return false;
    }
    if (this->bank != other.bank) {
      return false;
    }
    return true;
  }
  bool operator!=(const AddStaticTransformGps_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AddStaticTransformGps_Request_

// alias to use template instance with default allocator
using AddStaticTransformGps_Request =
  as2_msgs::srv::AddStaticTransformGps_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct AddStaticTransformGps_Response_
{
  using Type = AddStaticTransformGps_Response_<ContainerAllocator>;

  explicit AddStaticTransformGps_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit AddStaticTransformGps_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Response
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__srv__AddStaticTransformGps_Response
    std::shared_ptr<as2_msgs::srv::AddStaticTransformGps_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AddStaticTransformGps_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const AddStaticTransformGps_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AddStaticTransformGps_Response_

// alias to use template instance with default allocator
using AddStaticTransformGps_Response =
  as2_msgs::srv::AddStaticTransformGps_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace as2_msgs

namespace as2_msgs
{

namespace srv
{

struct AddStaticTransformGps
{
  using Request = as2_msgs::srv::AddStaticTransformGps_Request;
  using Response = as2_msgs::srv::AddStaticTransformGps_Response;
};

}  // namespace srv

}  // namespace as2_msgs

#endif  // AS2_MSGS__SRV__DETAIL__ADD_STATIC_TRANSFORM_GPS__STRUCT_HPP_
