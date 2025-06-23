// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from as2_msgs:action/Takeoff.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__TAKEOFF__STRUCT_HPP_
#define AS2_MSGS__ACTION__DETAIL__TAKEOFF__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_Goal __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_Goal __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_Goal_
{
  using Type = Takeoff_Goal_<ContainerAllocator>;

  explicit Takeoff_Goal_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->takeoff_height = 0.0f;
      this->takeoff_speed = 0.0f;
    }
  }

  explicit Takeoff_Goal_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->takeoff_height = 0.0f;
      this->takeoff_speed = 0.0f;
    }
  }

  // field types and members
  using _takeoff_height_type =
    float;
  _takeoff_height_type takeoff_height;
  using _takeoff_speed_type =
    float;
  _takeoff_speed_type takeoff_speed;

  // setters for named parameter idiom
  Type & set__takeoff_height(
    const float & _arg)
  {
    this->takeoff_height = _arg;
    return *this;
  }
  Type & set__takeoff_speed(
    const float & _arg)
  {
    this->takeoff_speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::Takeoff_Goal_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_Goal_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_Goal_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_Goal_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_Goal
    std::shared_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_Goal
    std::shared_ptr<as2_msgs::action::Takeoff_Goal_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_Goal_ & other) const
  {
    if (this->takeoff_height != other.takeoff_height) {
      return false;
    }
    if (this->takeoff_speed != other.takeoff_speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_Goal_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_Goal_

// alias to use template instance with default allocator
using Takeoff_Goal =
  as2_msgs::action::Takeoff_Goal_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_Result __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_Result __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_Result_
{
  using Type = Takeoff_Result_<ContainerAllocator>;

  explicit Takeoff_Result_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->takeoff_success = false;
    }
  }

  explicit Takeoff_Result_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->takeoff_success = false;
    }
  }

  // field types and members
  using _takeoff_success_type =
    bool;
  _takeoff_success_type takeoff_success;

  // setters for named parameter idiom
  Type & set__takeoff_success(
    const bool & _arg)
  {
    this->takeoff_success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::Takeoff_Result_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_Result_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_Result_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_Result_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_Result
    std::shared_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_Result
    std::shared_ptr<as2_msgs::action::Takeoff_Result_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_Result_ & other) const
  {
    if (this->takeoff_success != other.takeoff_success) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_Result_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_Result_

// alias to use template instance with default allocator
using Takeoff_Result =
  as2_msgs::action::Takeoff_Result_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_Feedback __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_Feedback __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_Feedback_
{
  using Type = Takeoff_Feedback_<ContainerAllocator>;

  explicit Takeoff_Feedback_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->actual_takeoff_speed = 0.0f;
      this->actual_takeoff_height = 0.0f;
    }
  }

  explicit Takeoff_Feedback_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->actual_takeoff_speed = 0.0f;
      this->actual_takeoff_height = 0.0f;
    }
  }

  // field types and members
  using _actual_takeoff_speed_type =
    float;
  _actual_takeoff_speed_type actual_takeoff_speed;
  using _actual_takeoff_height_type =
    float;
  _actual_takeoff_height_type actual_takeoff_height;

  // setters for named parameter idiom
  Type & set__actual_takeoff_speed(
    const float & _arg)
  {
    this->actual_takeoff_speed = _arg;
    return *this;
  }
  Type & set__actual_takeoff_height(
    const float & _arg)
  {
    this->actual_takeoff_height = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_Feedback
    std::shared_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_Feedback
    std::shared_ptr<as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_Feedback_ & other) const
  {
    if (this->actual_takeoff_speed != other.actual_takeoff_speed) {
      return false;
    }
    if (this->actual_takeoff_height != other.actual_takeoff_height) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_Feedback_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_Feedback_

// alias to use template instance with default allocator
using Takeoff_Feedback =
  as2_msgs::action::Takeoff_Feedback_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'goal'
#include "as2_msgs/action/detail/takeoff__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_SendGoal_Request_
{
  using Type = Takeoff_SendGoal_Request_<ContainerAllocator>;

  explicit Takeoff_SendGoal_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    goal(_init)
  {
    (void)_init;
  }

  explicit Takeoff_SendGoal_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::Takeoff_Goal_<ContainerAllocator>;
  _goal_type goal;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__goal(
    const as2_msgs::action::Takeoff_Goal_<ContainerAllocator> & _arg)
  {
    this->goal = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Request
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Request
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_SendGoal_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->goal != other.goal) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_SendGoal_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_SendGoal_Request_

// alias to use template instance with default allocator
using Takeoff_SendGoal_Request =
  as2_msgs::action::Takeoff_SendGoal_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_SendGoal_Response_
{
  using Type = Takeoff_SendGoal_Response_<ContainerAllocator>;

  explicit Takeoff_SendGoal_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->accepted = false;
    }
  }

  explicit Takeoff_SendGoal_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Response
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_SendGoal_Response
    std::shared_ptr<as2_msgs::action::Takeoff_SendGoal_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_SendGoal_Response_ & other) const
  {
    if (this->accepted != other.accepted) {
      return false;
    }
    if (this->stamp != other.stamp) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_SendGoal_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_SendGoal_Response_

// alias to use template instance with default allocator
using Takeoff_SendGoal_Response =
  as2_msgs::action::Takeoff_SendGoal_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

namespace as2_msgs
{

namespace action
{

struct Takeoff_SendGoal
{
  using Request = as2_msgs::action::Takeoff_SendGoal_Request;
  using Response = as2_msgs::action::Takeoff_SendGoal_Response;
};

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_GetResult_Request __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_GetResult_Request __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_GetResult_Request_
{
  using Type = Takeoff_GetResult_Request_<ContainerAllocator>;

  explicit Takeoff_GetResult_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init)
  {
    (void)_init;
  }

  explicit Takeoff_GetResult_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_GetResult_Request
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_GetResult_Request
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_GetResult_Request_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_GetResult_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_GetResult_Request_

// alias to use template instance with default allocator
using Takeoff_GetResult_Request =
  as2_msgs::action::Takeoff_GetResult_Request_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/takeoff__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_GetResult_Response __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_GetResult_Response __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_GetResult_Response_
{
  using Type = Takeoff_GetResult_Response_<ContainerAllocator>;

  explicit Takeoff_GetResult_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : result(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->status = 0;
    }
  }

  explicit Takeoff_GetResult_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::Takeoff_Result_<ContainerAllocator>;
  _result_type result;

  // setters for named parameter idiom
  Type & set__status(
    const int8_t & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__result(
    const as2_msgs::action::Takeoff_Result_<ContainerAllocator> & _arg)
  {
    this->result = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_GetResult_Response
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_GetResult_Response
    std::shared_ptr<as2_msgs::action::Takeoff_GetResult_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_GetResult_Response_ & other) const
  {
    if (this->status != other.status) {
      return false;
    }
    if (this->result != other.result) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_GetResult_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_GetResult_Response_

// alias to use template instance with default allocator
using Takeoff_GetResult_Response =
  as2_msgs::action::Takeoff_GetResult_Response_<std::allocator<void>>;

// constant definitions

}  // namespace action

}  // namespace as2_msgs

namespace as2_msgs
{

namespace action
{

struct Takeoff_GetResult
{
  using Request = as2_msgs::action::Takeoff_GetResult_Request;
  using Response = as2_msgs::action::Takeoff_GetResult_Response;
};

}  // namespace action

}  // namespace as2_msgs


// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.hpp"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/takeoff__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__as2_msgs__action__Takeoff_FeedbackMessage __attribute__((deprecated))
#else
# define DEPRECATED__as2_msgs__action__Takeoff_FeedbackMessage __declspec(deprecated)
#endif

namespace as2_msgs
{

namespace action
{

// message struct
template<class ContainerAllocator>
struct Takeoff_FeedbackMessage_
{
  using Type = Takeoff_FeedbackMessage_<ContainerAllocator>;

  explicit Takeoff_FeedbackMessage_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : goal_id(_init),
    feedback(_init)
  {
    (void)_init;
  }

  explicit Takeoff_FeedbackMessage_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    as2_msgs::action::Takeoff_Feedback_<ContainerAllocator>;
  _feedback_type feedback;

  // setters for named parameter idiom
  Type & set__goal_id(
    const unique_identifier_msgs::msg::UUID_<ContainerAllocator> & _arg)
  {
    this->goal_id = _arg;
    return *this;
  }
  Type & set__feedback(
    const as2_msgs::action::Takeoff_Feedback_<ContainerAllocator> & _arg)
  {
    this->feedback = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator> *;
  using ConstRawPtr =
    const as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__as2_msgs__action__Takeoff_FeedbackMessage
    std::shared_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__as2_msgs__action__Takeoff_FeedbackMessage
    std::shared_ptr<as2_msgs::action::Takeoff_FeedbackMessage_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Takeoff_FeedbackMessage_ & other) const
  {
    if (this->goal_id != other.goal_id) {
      return false;
    }
    if (this->feedback != other.feedback) {
      return false;
    }
    return true;
  }
  bool operator!=(const Takeoff_FeedbackMessage_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Takeoff_FeedbackMessage_

// alias to use template instance with default allocator
using Takeoff_FeedbackMessage =
  as2_msgs::action::Takeoff_FeedbackMessage_<std::allocator<void>>;

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

struct Takeoff
{
  /// The goal message defined in the action definition.
  using Goal = as2_msgs::action::Takeoff_Goal;
  /// The result message defined in the action definition.
  using Result = as2_msgs::action::Takeoff_Result;
  /// The feedback message defined in the action definition.
  using Feedback = as2_msgs::action::Takeoff_Feedback;

  struct Impl
  {
    /// The send_goal service using a wrapped version of the goal message as a request.
    using SendGoalService = as2_msgs::action::Takeoff_SendGoal;
    /// The get_result service using a wrapped version of the result message as a response.
    using GetResultService = as2_msgs::action::Takeoff_GetResult;
    /// The feedback message with generic fields which wraps the feedback message.
    using FeedbackMessage = as2_msgs::action::Takeoff_FeedbackMessage;

    /// The generic service to cancel a goal.
    using CancelGoalService = action_msgs::srv::CancelGoal;
    /// The generic message for the status of a goal.
    using GoalStatusMessage = action_msgs::msg::GoalStatusArray;
  };
};

typedef struct Takeoff Takeoff;

}  // namespace action

}  // namespace as2_msgs

#endif  // AS2_MSGS__ACTION__DETAIL__TAKEOFF__STRUCT_HPP_
