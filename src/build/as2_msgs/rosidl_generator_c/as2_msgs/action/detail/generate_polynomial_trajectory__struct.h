// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:action/GeneratePolynomialTrajectory.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__GENERATE_POLYNOMIAL_TRAJECTORY__STRUCT_H_
#define AS2_MSGS__ACTION__DETAIL__GENERATE_POLYNOMIAL_TRAJECTORY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'yaw'
#include "as2_msgs/msg/detail/yaw_mode__struct.h"
// Member 'path'
#include "as2_msgs/msg/detail/pose_with_id__struct.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_Goal
{
  /// Request
  /// Message header, with the frame of the pose list
  std_msgs__msg__Header header;
  /// Yaw goal mode
  as2_msgs__msg__YawMode yaw;
  /// List of poses with ID in path
  as2_msgs__msg__PoseWithID__Sequence path;
  /// Maximum speed desired in path (m/s)
  float max_speed;
} as2_msgs__action__GeneratePolynomialTrajectory_Goal;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_Goal.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_Goal__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_Result
{
  /// False if failed to follow the generated trajectory
  bool trajectory_generator_success;
} as2_msgs__action__GeneratePolynomialTrajectory_Result;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_Result.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_Result__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_Result__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'next_waypoint_id'
#include "rosidl_runtime_c/string.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_Feedback
{
  /// Next waypoint id in path to follow
  rosidl_runtime_c__String next_waypoint_id;
  /// Number of remaining waypoints to follow
  uint16_t remaining_waypoints;
} as2_msgs__action__GeneratePolynomialTrajectory_Feedback;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_Feedback.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_Feedback__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__GeneratePolynomialTrajectory_Goal goal;
} as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Request;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Request.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Request__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Response;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Response.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Response__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Request;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Request.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Request__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Response
{
  int8_t status;
  as2_msgs__action__GeneratePolynomialTrajectory_Result result;
} as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Response;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Response.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Response__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/generate_polynomial_trajectory__struct.h"

/// Struct defined in action/GeneratePolynomialTrajectory in the package as2_msgs.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__GeneratePolynomialTrajectory_Feedback feedback;
} as2_msgs__action__GeneratePolynomialTrajectory_FeedbackMessage;

// Struct for a sequence of as2_msgs__action__GeneratePolynomialTrajectory_FeedbackMessage.
typedef struct as2_msgs__action__GeneratePolynomialTrajectory_FeedbackMessage__Sequence
{
  as2_msgs__action__GeneratePolynomialTrajectory_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GeneratePolynomialTrajectory_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__ACTION__DETAIL__GENERATE_POLYNOMIAL_TRAJECTORY__STRUCT_H_
