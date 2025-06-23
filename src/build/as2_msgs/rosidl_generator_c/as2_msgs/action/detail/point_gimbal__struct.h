// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:action/PointGimbal.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__STRUCT_H_
#define AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'control'
#include "as2_msgs/msg/detail/gimbal_control__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_Goal
{
  /// Request
  /// Goal target
  as2_msgs__msg__GimbalControl control;
  /// Keep following after reach target
  bool follow_mode;
} as2_msgs__action__PointGimbal_Goal;

// Struct for a sequence of as2_msgs__action__PointGimbal_Goal.
typedef struct as2_msgs__action__PointGimbal_Goal__Sequence
{
  as2_msgs__action__PointGimbal_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_Result
{
  /// False if failed to point to target
  bool success;
} as2_msgs__action__PointGimbal_Result;

// Struct for a sequence of as2_msgs__action__PointGimbal_Result.
typedef struct as2_msgs__action__PointGimbal_Result__Sequence
{
  as2_msgs__action__PointGimbal_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_Result__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'gimbal_attitude'
// Member 'gimbal_speed'
#include "geometry_msgs/msg/detail/vector3_stamped__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_Feedback
{
  /// Current attitude (rad)
  geometry_msgs__msg__Vector3Stamped gimbal_attitude;
  /// Current speed (rad/s)
  geometry_msgs__msg__Vector3Stamped gimbal_speed;
} as2_msgs__action__PointGimbal_Feedback;

// Struct for a sequence of as2_msgs__action__PointGimbal_Feedback.
typedef struct as2_msgs__action__PointGimbal_Feedback__Sequence
{
  as2_msgs__action__PointGimbal_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "as2_msgs/action/detail/point_gimbal__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__PointGimbal_Goal goal;
} as2_msgs__action__PointGimbal_SendGoal_Request;

// Struct for a sequence of as2_msgs__action__PointGimbal_SendGoal_Request.
typedef struct as2_msgs__action__PointGimbal_SendGoal_Request__Sequence
{
  as2_msgs__action__PointGimbal_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} as2_msgs__action__PointGimbal_SendGoal_Response;

// Struct for a sequence of as2_msgs__action__PointGimbal_SendGoal_Response.
typedef struct as2_msgs__action__PointGimbal_SendGoal_Response__Sequence
{
  as2_msgs__action__PointGimbal_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} as2_msgs__action__PointGimbal_GetResult_Request;

// Struct for a sequence of as2_msgs__action__PointGimbal_GetResult_Request.
typedef struct as2_msgs__action__PointGimbal_GetResult_Request__Sequence
{
  as2_msgs__action__PointGimbal_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/point_gimbal__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_GetResult_Response
{
  int8_t status;
  as2_msgs__action__PointGimbal_Result result;
} as2_msgs__action__PointGimbal_GetResult_Response;

// Struct for a sequence of as2_msgs__action__PointGimbal_GetResult_Response.
typedef struct as2_msgs__action__PointGimbal_GetResult_Response__Sequence
{
  as2_msgs__action__PointGimbal_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/point_gimbal__struct.h"

/// Struct defined in action/PointGimbal in the package as2_msgs.
typedef struct as2_msgs__action__PointGimbal_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__PointGimbal_Feedback feedback;
} as2_msgs__action__PointGimbal_FeedbackMessage;

// Struct for a sequence of as2_msgs__action__PointGimbal_FeedbackMessage.
typedef struct as2_msgs__action__PointGimbal_FeedbackMessage__Sequence
{
  as2_msgs__action__PointGimbal_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__PointGimbal_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__ACTION__DETAIL__POINT_GIMBAL__STRUCT_H_
