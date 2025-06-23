// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:action/NavigateToPoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__STRUCT_H_
#define AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'point'
#include "geometry_msgs/msg/detail/point_stamped__struct.h"
// Member 'yaw'
#include "as2_msgs/msg/detail/yaw_mode__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_Goal
{
  /// Request
  /// Goal pose 3D (m)
  geometry_msgs__msg__PointStamped point;
  /// Yaw goal mode
  as2_msgs__msg__YawMode yaw;
  /// Maximum speed desired in path (m/s)
  float navigation_speed;
} as2_msgs__action__NavigateToPoint_Goal;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_Goal.
typedef struct as2_msgs__action__NavigateToPoint_Goal__Sequence
{
  as2_msgs__action__NavigateToPoint_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_Result
{
  /// Point reached?
  bool success;
} as2_msgs__action__NavigateToPoint_Result;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_Result.
typedef struct as2_msgs__action__NavigateToPoint_Result__Sequence
{
  as2_msgs__action__NavigateToPoint_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_Result__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'current_pose'
#include "geometry_msgs/msg/detail/pose_stamped__struct.h"
// Member 'current_speed'
#include "geometry_msgs/msg/detail/twist_stamped__struct.h"
// Member 'navigation_time'
// Member 'estimated_time_remaining'
#include "builtin_interfaces/msg/detail/duration__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_Feedback
{
  /// Current pose (m)
  geometry_msgs__msg__PoseStamped current_pose;
  /// Current speed (m/s)
  geometry_msgs__msg__TwistStamped current_speed;
  /// Time from departure (s)
  builtin_interfaces__msg__Duration navigation_time;
  /// Time to goal (s)
  builtin_interfaces__msg__Duration estimated_time_remaining;
  /// Distance to goal (m)
  float distance_remaining;
} as2_msgs__action__NavigateToPoint_Feedback;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_Feedback.
typedef struct as2_msgs__action__NavigateToPoint_Feedback__Sequence
{
  as2_msgs__action__NavigateToPoint_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "as2_msgs/action/detail/navigate_to_point__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__NavigateToPoint_Goal goal;
} as2_msgs__action__NavigateToPoint_SendGoal_Request;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_SendGoal_Request.
typedef struct as2_msgs__action__NavigateToPoint_SendGoal_Request__Sequence
{
  as2_msgs__action__NavigateToPoint_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} as2_msgs__action__NavigateToPoint_SendGoal_Response;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_SendGoal_Response.
typedef struct as2_msgs__action__NavigateToPoint_SendGoal_Response__Sequence
{
  as2_msgs__action__NavigateToPoint_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} as2_msgs__action__NavigateToPoint_GetResult_Request;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_GetResult_Request.
typedef struct as2_msgs__action__NavigateToPoint_GetResult_Request__Sequence
{
  as2_msgs__action__NavigateToPoint_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/navigate_to_point__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_GetResult_Response
{
  int8_t status;
  as2_msgs__action__NavigateToPoint_Result result;
} as2_msgs__action__NavigateToPoint_GetResult_Response;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_GetResult_Response.
typedef struct as2_msgs__action__NavigateToPoint_GetResult_Response__Sequence
{
  as2_msgs__action__NavigateToPoint_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/navigate_to_point__struct.h"

/// Struct defined in action/NavigateToPoint in the package as2_msgs.
typedef struct as2_msgs__action__NavigateToPoint_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__NavigateToPoint_Feedback feedback;
} as2_msgs__action__NavigateToPoint_FeedbackMessage;

// Struct for a sequence of as2_msgs__action__NavigateToPoint_FeedbackMessage.
typedef struct as2_msgs__action__NavigateToPoint_FeedbackMessage__Sequence
{
  as2_msgs__action__NavigateToPoint_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__NavigateToPoint_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__ACTION__DETAIL__NAVIGATE_TO_POINT__STRUCT_H_
