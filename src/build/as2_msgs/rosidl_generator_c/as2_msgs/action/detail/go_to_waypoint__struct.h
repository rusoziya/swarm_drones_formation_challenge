// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:action/GoToWaypoint.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__GO_TO_WAYPOINT__STRUCT_H_
#define AS2_MSGS__ACTION__DETAIL__GO_TO_WAYPOINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'yaw'
#include "as2_msgs/msg/detail/yaw_mode__struct.h"
// Member 'target_pose'
#include "geometry_msgs/msg/detail/point_stamped__struct.h"

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_Goal
{
  /// Request
  /// Yaw mode
  as2_msgs__msg__YawMode yaw;
  /// Goal pose 3D (m)
  geometry_msgs__msg__PointStamped target_pose;
  /// Maximum speed (m/s)
  float max_speed;
} as2_msgs__action__GoToWaypoint_Goal;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_Goal.
typedef struct as2_msgs__action__GoToWaypoint_Goal__Sequence
{
  as2_msgs__action__GoToWaypoint_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_Result
{
  /// False if failed to takeoff
  bool go_to_success;
} as2_msgs__action__GoToWaypoint_Result;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_Result.
typedef struct as2_msgs__action__GoToWaypoint_Result__Sequence
{
  as2_msgs__action__GoToWaypoint_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_Result__Sequence;


// Constants defined in the message

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_Feedback
{
  /// Actual speed (m/s)
  float actual_speed;
  /// Distance to goal (m)
  float actual_distance_to_goal;
} as2_msgs__action__GoToWaypoint_Feedback;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_Feedback.
typedef struct as2_msgs__action__GoToWaypoint_Feedback__Sequence
{
  as2_msgs__action__GoToWaypoint_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "as2_msgs/action/detail/go_to_waypoint__struct.h"

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__GoToWaypoint_Goal goal;
} as2_msgs__action__GoToWaypoint_SendGoal_Request;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_SendGoal_Request.
typedef struct as2_msgs__action__GoToWaypoint_SendGoal_Request__Sequence
{
  as2_msgs__action__GoToWaypoint_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} as2_msgs__action__GoToWaypoint_SendGoal_Response;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_SendGoal_Response.
typedef struct as2_msgs__action__GoToWaypoint_SendGoal_Response__Sequence
{
  as2_msgs__action__GoToWaypoint_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} as2_msgs__action__GoToWaypoint_GetResult_Request;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_GetResult_Request.
typedef struct as2_msgs__action__GoToWaypoint_GetResult_Request__Sequence
{
  as2_msgs__action__GoToWaypoint_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/go_to_waypoint__struct.h"

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_GetResult_Response
{
  int8_t status;
  as2_msgs__action__GoToWaypoint_Result result;
} as2_msgs__action__GoToWaypoint_GetResult_Response;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_GetResult_Response.
typedef struct as2_msgs__action__GoToWaypoint_GetResult_Response__Sequence
{
  as2_msgs__action__GoToWaypoint_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/go_to_waypoint__struct.h"

/// Struct defined in action/GoToWaypoint in the package as2_msgs.
typedef struct as2_msgs__action__GoToWaypoint_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__GoToWaypoint_Feedback feedback;
} as2_msgs__action__GoToWaypoint_FeedbackMessage;

// Struct for a sequence of as2_msgs__action__GoToWaypoint_FeedbackMessage.
typedef struct as2_msgs__action__GoToWaypoint_FeedbackMessage__Sequence
{
  as2_msgs__action__GoToWaypoint_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__GoToWaypoint_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__ACTION__DETAIL__GO_TO_WAYPOINT__STRUCT_H_
