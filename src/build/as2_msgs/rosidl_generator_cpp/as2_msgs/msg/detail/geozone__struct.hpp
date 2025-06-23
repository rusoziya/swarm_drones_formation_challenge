// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:msg/Geozone.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__GEOZONE__STRUCT_HPP_
#define AS2_MSGS__MSG__DETAIL__GEOZONE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'polygon'
#include "geometry_msgs/msg/detail/polygon__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__msg__Geozone __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__msg__Geozone __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Geozone_
{
  using Type = Geozone_<ContainerAllocator>;

  explicit Geozone_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : polygon(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->alert = 0;
      this->type = "";
      this->data_type = "";
      this->z_up = 0.0f;
      this->z_down = 0.0f;
    }
  }

  explicit Geozone_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : type(_alloc),
    data_type(_alloc),
    polygon(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->alert = 0;
      this->type = "";
      this->data_type = "";
      this->z_up = 0.0f;
      this->z_down = 0.0f;
    }
  }

  // field types and members
  using _id_type =
    int8_t;
  _id_type id;
  using _alert_type =
    int8_t;
  _alert_type alert;
  using _type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _type_type type;
  using _data_type_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _data_type_type data_type;
  using _polygon_type =
    geometry_msgs::msg::Polygon_<ContainerAllocator>;
  _polygon_type polygon;
  using _z_up_type =
    float;
  _z_up_type z_up;
  using _z_down_type =
    float;
  _z_down_type z_down;

  // setters for named parameter idiom
  Type & set__id(
    const int8_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__alert(
    const int8_t & _arg)
  {
    this->alert = _arg;
    return *this;
  }
  Type & set__type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__data_type(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->data_type = _arg;
    return *this;
  }
  Type & set__polygon(
    const geometry_msgs::msg::Polygon_<ContainerAllocator> & _arg)
  {
    this->polygon = _arg;
    return *this;
  }
  Type & set__z_up(
    const float & _arg)
  {
    this->z_up = _arg;
    return *this;
  }
  Type & set__z_down(
    const float & _arg)
  {
    this->z_down = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::msg::Geozone_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::msg::Geozone_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::msg::Geozone_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::msg::Geozone_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::Geozone_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::Geozone_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::msg::Geozone_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::msg::Geozone_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::msg::Geozone_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::msg::Geozone_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__msg__Geozone
    std::shared_ptr<as2_msgs::msg::Geozone_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__msg__Geozone
    std::shared_ptr<as2_msgs::msg::Geozone_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Geozone_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->alert != other.alert) {
      return false;
    }
    if (this->type != other.type) {
      return false;
    }
    if (this->data_type != other.data_type) {
      return false;
    }
    if (this->polygon != other.polygon) {
      return false;
    }
    if (this->z_up != other.z_up) {
      return false;
    }
    if (this->z_down != other.z_down) {
      return false;
    }
    return true;
  }
  bool operator!=(const Geozone_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Geozone_

// alias to use template instance with default allocator
using Geozone =
  as2_msgs::msg::Geozone_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace as2_msgs

#endif  // AS2_MSGS__MSG__DETAIL__GEOZONE__STRUCT_HPP_
