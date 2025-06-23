// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:action/PointGimbal.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__STRUCT_HPP_
#define AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'control'
#include "as2_msgs/msg/detail/gimbal_control__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_Goal __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_Goal __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_Goal_
{
  using Type = PointGimbal_Goal_<ContainerAllocator>;

  explicit PointGimbal_Goal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : control(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->follow_mode = false;
    }
  }

  explicit PointGimbal_Goal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : control(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->follow_mode = false;
    }
  }

  // field types and members
  using _control_type =
    as2_msgs::msg::GimbalControl_<ContainerAllocator>;
  _control_type control;
  using _follow_mode_type =
    bool;
  _follow_mode_type follow_mode;

  // setters for named parameter idiom
  Type & set__control(
    const as2_msgs::msg::GimbalControl_<ContainerAllocator> & _arg)
  {
    this->control = _arg;
    return *this;
  }
  Type & set__follow_mode(
    const bool & _arg)
  {
    this->follow_mode = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_Goal
    std::shared_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_Goal
    std::shared_ptr<as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_Goal_ & other) const
  {
    if (this->control != other.control) {
      return false;
    }
    if (this->follow_mode != other.follow_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_Goal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_Goal_

// alias to use template instance with default allocator
using PointGimbal_Goal =
  as2_msgs::action::PointGimbal_Goal_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_Result __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_Result __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_Result_
{
  using Type = PointGimbal_Result_<ContainerAllocator>;

  explicit PointGimbal_Result_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit PointGimbal_Result_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::PointGimbal_Result_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_Result_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_Result_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_Result_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_Result
    std::shared_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_Result
    std::shared_ptr<as2_msgs::action::PointGimbal_Result_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_Result_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_Result_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_Result_

// alias to use template instance with default allocator
using PointGimbal_Result =
  as2_msgs::action::PointGimbal_Result_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'gimbal_attitude'
// Member 'gimbal_speed'
#include "geometry_msgs/msg/detail/vector3_stamped__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_Feedback __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_Feedback __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_Feedback_
{
  using Type = PointGimbal_Feedback_<ContainerAllocator>;

  explicit PointGimbal_Feedback_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : gimbal_attitude(_init),
    gimbal_speed(_init)
  {
    (void)_init;
  }

  explicit PointGimbal_Feedback_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : gimbal_attitude(_alloc, _init),
    gimbal_speed(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _gimbal_attitude_type =
    geometry_msgs::msg::Vector3Stamped_<ContainerAllocator>;
  _gimbal_attitude_type gimbal_attitude;
  using _gimbal_speed_type =
    geometry_msgs::msg::Vector3Stamped_<ContainerAllocator>;
  _gimbal_speed_type gimbal_speed;

  // setters for named parameter idiom
  Type & set__gimbal_attitude(
    const geometry_msgs::msg::Vector3Stamped_<ContainerAllocator> & _arg)
  {
    this->gimbal_attitude = _arg;
    return *this;
  }
  Type & set__gimbal_speed(
    const geometry_msgs::msg::Vector3Stamped_<ContainerAllocator> & _arg)
  {
    this->gimbal_speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_Feedback
    std::shared_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_Feedback
    std::shared_ptr<as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_Feedback_ & other) const
  {
    if (this->gimbal_attitude != other.gimbal_attitude) {
      return false;
    }
    if (this->gimbal_speed != other.gimbal_speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_Feedback_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_Feedback_

// alias to use template instance with default allocator
using PointGimbal_Feedback =
  as2_msgs::action::PointGimbal_Feedback_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'goal'
#include "as2_msgs/action/detail/point_gimbal__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_SendGoal_Request_
{
  using Type = PointGimbal_SendGoal_Request_<ContainerAllocator>;

  explicit PointGimbal_SendGoal_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    goal(_init)
  {
    (void)_init;
  }

  explicit PointGimbal_SendGoal_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_alloc, _init),
    goal(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _goal_id_type =
    unique_identifier_msgs::msg::UUID_<ContainerAllocator>;
  _goal_id_type goal_id;
  using _goal_type =
    as2_msgs::action::PointGimbal_Goal_<ContainerAllocator>;
  _goal_type goal;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__goal(
    const as2_msgs::action::PointGimbal_Goal_<ContainerAllocator> & _arg)
  {
    this->goal = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Request
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Request
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_SendGoal_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->goal != other.goal) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_SendGoal_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_SendGoal_Request_

// alias to use template instance with default allocator
using PointGimbal_SendGoal_Request =
  as2_msgs::action::PointGimbal_SendGoal_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_SendGoal_Response_
{
  using Type = PointGimbal_SendGoal_Response_<ContainerAllocator>;

  explicit PointGimbal_SendGoal_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
    }
  }

  explicit PointGimbal_SendGoal_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
    }
  }

  // field types and members
  using _accepted_type =
    bool;
  _accepted_type accepted;
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;

  // setters for named parameter idiom
  Type & set__accepted(
    const bool & _arg)
  {
    this->accepted = _arg;
    return *this;
  }
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Response
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_SendGoal_Response
    std::shared_ptr<as2_msgs::action::PointGimbal_SendGoal_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_SendGoal_Response_ & other) const
  {
    if (this->accepted != other.accepted) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_SendGoal_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_SendGoal_Response_

// alias to use template instance with default allocator
using PointGimbal_SendGoal_Response =
  as2_msgs::action::PointGimbal_SendGoal_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

namespace as2_msgs
{

namespace action
{

struct PointGimbal_SendGoal
{
  using Request = as2_msgs::action::PointGimbal_SendGoal_Request;
  using Response = as2_msgs::action::PointGimbal_SendGoal_Response;
};

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_GetResult_Request_
{
  using Type = PointGimbal_GetResult_Request_<ContainerAllocator>;

  explicit PointGimbal_GetResult_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init)
  {
    (void)_init;
  }

  explicit PointGimbal_GetResult_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _goal_id_type =
    unique_identifier_msgs::msg::UUID_<ContainerAllocator>;
  _goal_id_type goal_id;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Request
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Request
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_GetResult_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_GetResult_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_GetResult_Request_

// alias to use template instance with default allocator
using PointGimbal_GetResult_Request =
  as2_msgs::action::PointGimbal_GetResult_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/point_gimbal__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_GetResult_Response_
{
  using Type = PointGimbal_GetResult_Response_<ContainerAllocator>;

  explicit PointGimbal_GetResult_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  explicit PointGimbal_GetResult_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  // field types and members
  using _status_type =
    int8_t;
  _status_type status;
  using _result_type =
    as2_msgs::action::PointGimbal_Result_<ContainerAllocator>;
  _result_type result;

  // setters for named parameter idiom
  Type & set__status(
    const int8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__result(
    const as2_msgs::action::PointGimbal_Result_<ContainerAllocator> & _arg)
  {
    this->result = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Response
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_GetResult_Response
    std::shared_ptr<as2_msgs::action::PointGimbal_GetResult_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_GetResult_Response_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    if (this->result != other.result) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_GetResult_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_GetResult_Response_

// alias to use template instance with default allocator
using PointGimbal_GetResult_Response =
  as2_msgs::action::PointGimbal_GetResult_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

namespace as2_msgs
{

namespace action
{

struct PointGimbal_GetResult
{
  using Request = as2_msgs::action::PointGimbal_GetResult_Request;
  using Response = as2_msgs::action::PointGimbal_GetResult_Response;
};

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/point_gimbal__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__PointGimbal_FeedbackMessage __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__PointGimbal_FeedbackMessage __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct PointGimbal_FeedbackMessage_
{
  using Type = PointGimbal_FeedbackMessage_<ContainerAllocator>;

  explicit PointGimbal_FeedbackMessage_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    feedback(_init)
  {
    (void)_init;
  }

  explicit PointGimbal_FeedbackMessage_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_alloc, _init),
    feedback(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _goal_id_type =
    unique_identifier_msgs::msg::UUID_<ContainerAllocator>;
  _goal_id_type goal_id;
  using _feedback_type =
    as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator>;
  _feedback_type feedback;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__feedback(
    const as2_msgs::action::PointGimbal_Feedback_<ContainerAllocator> & _arg)
  {
    this->feedback = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__PointGimbal_FeedbackMessage
    std::shared_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__PointGimbal_FeedbackMessage
    std::shared_ptr<as2_msgs::action::PointGimbal_FeedbackMessage_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PointGimbal_FeedbackMessage_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->feedback != other.feedback) {
      return false;
    }
    return true;
  }
  bool operator!=(const PointGimbal_FeedbackMessage_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PointGimbal_FeedbackMessage_

// alias to use template instance with default allocator
using PointGimbal_FeedbackMessage =
  as2_msgs::action::PointGimbal_FeedbackMessage_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

#include "action_msgs/srv/cancel_goal.hpp"
#include "action_msgs/msg/goal_info.hpp"
#include "action_msgs/msg/goal_status_array.hpp"

namespace as2_msgs
{

namespace action
{

struct PointGimbal
{
  /// The goal message defined in the action definition.
  using Goal = as2_msgs::action::PointGimbal_Goal;
  /// The result message defined in the action definition.
  using Result = as2_msgs::action::PointGimbal_Result;
  /// The feedback message defined in the action definition.
  using Feedback = as2_msgs::action::PointGimbal_Feedback;

  struct Impl
  {
    /// The send_goal service using a wrapped version of the goal message as a request.
    using SendGoalService = as2_msgs::action::PointGimbal_SendGoal;
    /// The get_result service using a wrapped version of the result message as a response.
    using GetResultService = as2_msgs::action::PointGimbal_GetResult;
    /// The feedback message with generic fields which wraps the feedback message.
    using FeedbackMessage = as2_msgs::action::PointGimbal_FeedbackMessage;

    /// The generic service to cancel a goal.
    using CancelGoalService = action_msgs::srv::CancelGoal;
    /// The generic message for the status of a goal.
    using GoalStatusMessage = action_msgs::msg::GoalStatusArray;
  };
};

typedef struct PointGimbal PointGimbal;

}  // namespace action

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__STRUCT_HPP_
