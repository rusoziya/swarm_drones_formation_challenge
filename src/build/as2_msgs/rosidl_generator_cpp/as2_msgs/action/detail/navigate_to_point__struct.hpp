// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:action/NavigateToPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__STRUCT_HPP_
#define AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'point'
#include "geometry_msgs/msg/detail/point_stamped__struct.hpp"
// Member 'yaw'
#include "as2_msgs/msg/detail/yaw_mode__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_Goal __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_Goal __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_Goal_
{
  using Type = NavigateToPoint_Goal_<ContainerAllocator>;

  explicit NavigateToPoint_Goal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : point(_init),
    yaw(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->navigation_speed = 0.0f;
    }
  }

  explicit NavigateToPoint_Goal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : point(_alloc, _init),
    yaw(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->navigation_speed = 0.0f;
    }
  }

  // field types and members
  using _point_type =
    geometry_msgs::msg::PointStamped_<ContainerAllocator>;
  _point_type point;
  using _yaw_type =
    as2_msgs::msg::YawMode_<ContainerAllocator>;
  _yaw_type yaw;
  using _navigation_speed_type =
    float;
  _navigation_speed_type navigation_speed;

  // setters for named parameter idiom
  Type & set__point(
    const geometry_msgs::msg::PointStamped_<ContainerAllocator> & _arg)
  {
    this->point = _arg;
    return *this;
  }
  Type & set__yaw(
    const as2_msgs::msg::YawMode_<ContainerAllocator> & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__navigation_speed(
    const float & _arg)
  {
    this->navigation_speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_Goal
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_Goal
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_Goal_ & other) const
  {
    if (this->point != other.point) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->navigation_speed != other.navigation_speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_Goal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_Goal_

// alias to use template instance with default allocator
using NavigateToPoint_Goal =
  as2_msgs::action::NavigateToPoint_Goal_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_Result __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_Result __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_Result_
{
  using Type = NavigateToPoint_Result_<ContainerAllocator>;

  explicit NavigateToPoint_Result_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit NavigateToPoint_Result_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_Result
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_Result
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_Result_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_Result_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_Result_

// alias to use template instance with default allocator
using NavigateToPoint_Result =
  as2_msgs::action::NavigateToPoint_Result_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose_stamped__struct.hpp"
// Member 'current_speed'
#include "geometry_msgs/msg/detail/twist_stamped__struct.hpp"
// Member 'navigation_time'
// Member 'estimated_time_remaining'
#include "builtin_interfaces/msg/detail/duration__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_Feedback __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_Feedback __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_Feedback_
{
  using Type = NavigateToPoint_Feedback_<ContainerAllocator>;

  explicit NavigateToPoint_Feedback_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : current_pose(_init),
    current_speed(_init),
    navigation_time(_init),
    estimated_time_remaining(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->distance_remaining = 0.0f;
    }
  }

  explicit NavigateToPoint_Feedback_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : current_pose(_alloc, _init),
    current_speed(_alloc, _init),
    navigation_time(_alloc, _init),
    estimated_time_remaining(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->distance_remaining = 0.0f;
    }
  }

  // field types and members
  using _current_pose_type =
    geometry_msgs::msg::PoseStamped_<ContainerAllocator>;
  _current_pose_type current_pose;
  using _current_speed_type =
    geometry_msgs::msg::TwistStamped_<ContainerAllocator>;
  _current_speed_type current_speed;
  using _navigation_time_type =
    builtin_interfaces::msg::Duration_<ContainerAllocator>;
  _navigation_time_type navigation_time;
  using _estimated_time_remaining_type =
    builtin_interfaces::msg::Duration_<ContainerAllocator>;
  _estimated_time_remaining_type estimated_time_remaining;
  using _distance_remaining_type =
    float;
  _distance_remaining_type distance_remaining;

  // setters for named parameter idiom
  Type & set__current_pose(
    const geometry_msgs::msg::PoseStamped_<ContainerAllocator> & _arg)
  {
    this->current_pose = _arg;
    return *this;
  }
  Type & set__current_speed(
    const geometry_msgs::msg::TwistStamped_<ContainerAllocator> & _arg)
  {
    this->current_speed = _arg;
    return *this;
  }
  Type & set__navigation_time(
    const builtin_interfaces::msg::Duration_<ContainerAllocator> & _arg)
  {
    this->navigation_time = _arg;
    return *this;
  }
  Type & set__estimated_time_remaining(
    const builtin_interfaces::msg::Duration_<ContainerAllocator> & _arg)
  {
    this->estimated_time_remaining = _arg;
    return *this;
  }
  Type & set__distance_remaining(
    const float & _arg)
  {
    this->distance_remaining = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_Feedback
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_Feedback
    std::shared_ptr<as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_Feedback_ & other) const
  {
    if (this->current_pose != other.current_pose) {
      return false;
    }
    if (this->current_speed != other.current_speed) {
      return false;
    }
    if (this->navigation_time != other.navigation_time) {
      return false;
    }
    if (this->estimated_time_remaining != other.estimated_time_remaining) {
      return false;
    }
    if (this->distance_remaining != other.distance_remaining) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_Feedback_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_Feedback_

// alias to use template instance with default allocator
using NavigateToPoint_Feedback =
  as2_msgs::action::NavigateToPoint_Feedback_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'goal'
#include "as2_msgs/action/detail/navigate_to_point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_SendGoal_Request_
{
  using Type = NavigateToPoint_SendGoal_Request_<ContainerAllocator>;

  explicit NavigateToPoint_SendGoal_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    goal(_init)
  {
    (void)_init;
  }

  explicit NavigateToPoint_SendGoal_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator>;
  _goal_type goal;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__goal(
    const as2_msgs::action::NavigateToPoint_Goal_<ContainerAllocator> & _arg)
  {
    this->goal = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Request
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Request
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_SendGoal_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->goal != other.goal) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_SendGoal_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_SendGoal_Request_

// alias to use template instance with default allocator
using NavigateToPoint_SendGoal_Request =
  as2_msgs::action::NavigateToPoint_SendGoal_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_SendGoal_Response_
{
  using Type = NavigateToPoint_SendGoal_Response_<ContainerAllocator>;

  explicit NavigateToPoint_SendGoal_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
    }
  }

  explicit NavigateToPoint_SendGoal_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Response
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_SendGoal_Response
    std::shared_ptr<as2_msgs::action::NavigateToPoint_SendGoal_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_SendGoal_Response_ & other) const
  {
    if (this->accepted != other.accepted) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_SendGoal_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_SendGoal_Response_

// alias to use template instance with default allocator
using NavigateToPoint_SendGoal_Response =
  as2_msgs::action::NavigateToPoint_SendGoal_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

namespace as2_msgs
{

namespace action
{

struct NavigateToPoint_SendGoal
{
  using Request = as2_msgs::action::NavigateToPoint_SendGoal_Request;
  using Response = as2_msgs::action::NavigateToPoint_SendGoal_Response;
};

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_GetResult_Request_
{
  using Type = NavigateToPoint_GetResult_Request_<ContainerAllocator>;

  explicit NavigateToPoint_GetResult_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init)
  {
    (void)_init;
  }

  explicit NavigateToPoint_GetResult_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Request
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Request
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_GetResult_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_GetResult_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_GetResult_Request_

// alias to use template instance with default allocator
using NavigateToPoint_GetResult_Request =
  as2_msgs::action::NavigateToPoint_GetResult_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/navigate_to_point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_GetResult_Response_
{
  using Type = NavigateToPoint_GetResult_Response_<ContainerAllocator>;

  explicit NavigateToPoint_GetResult_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  explicit NavigateToPoint_GetResult_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator>;
  _result_type result;

  // setters for named parameter idiom
  Type & set__status(
    const int8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__result(
    const as2_msgs::action::NavigateToPoint_Result_<ContainerAllocator> & _arg)
  {
    this->result = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Response
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_GetResult_Response
    std::shared_ptr<as2_msgs::action::NavigateToPoint_GetResult_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_GetResult_Response_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    if (this->result != other.result) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_GetResult_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_GetResult_Response_

// alias to use template instance with default allocator
using NavigateToPoint_GetResult_Response =
  as2_msgs::action::NavigateToPoint_GetResult_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

namespace as2_msgs
{

namespace action
{

struct NavigateToPoint_GetResult
{
  using Request = as2_msgs::action::NavigateToPoint_GetResult_Request;
  using Response = as2_msgs::action::NavigateToPoint_GetResult_Response;
};

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/navigate_to_point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__NavigateToPoint_FeedbackMessage __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__NavigateToPoint_FeedbackMessage __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct NavigateToPoint_FeedbackMessage_
{
  using Type = NavigateToPoint_FeedbackMessage_<ContainerAllocator>;

  explicit NavigateToPoint_FeedbackMessage_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    feedback(_init)
  {
    (void)_init;
  }

  explicit NavigateToPoint_FeedbackMessage_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator>;
  _feedback_type feedback;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__feedback(
    const as2_msgs::action::NavigateToPoint_Feedback_<ContainerAllocator> & _arg)
  {
    this->feedback = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_FeedbackMessage
    std::shared_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__NavigateToPoint_FeedbackMessage
    std::shared_ptr<as2_msgs::action::NavigateToPoint_FeedbackMessage_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigateToPoint_FeedbackMessage_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->feedback != other.feedback) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigateToPoint_FeedbackMessage_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigateToPoint_FeedbackMessage_

// alias to use template instance with default allocator
using NavigateToPoint_FeedbackMessage =
  as2_msgs::action::NavigateToPoint_FeedbackMessage_<std::allocator<void>>;

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

struct NavigateToPoint
{
  /// The goal message defined in the action definition.
  using Goal = as2_msgs::action::NavigateToPoint_Goal;
  /// The result message defined in the action definition.
  using Result = as2_msgs::action::NavigateToPoint_Result;
  /// The feedback message defined in the action definition.
  using Feedback = as2_msgs::action::NavigateToPoint_Feedback;

  struct Impl
  {
    /// The send_goal service using a wrapped version of the goal message as a request.
    using SendGoalService = as2_msgs::action::NavigateToPoint_SendGoal;
    /// The get_result service using a wrapped version of the result message as a response.
    using GetResultService = as2_msgs::action::NavigateToPoint_GetResult;
    /// The feedback message with generic fields which wraps the feedback message.
    using FeedbackMessage = as2_msgs::action::NavigateToPoint_FeedbackMessage;

    /// The generic service to cancel a goal.
    using CancelGoalService = action_msgs::srv::CancelGoal;
    /// The generic message for the status of a goal.
    using GoalStatusMessage = action_msgs::msg::GoalStatusArray;
  };
};

typedef struct NavigateToPoint NavigateToPoint;

}  // namespace action

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__STRUCT_HPP_
