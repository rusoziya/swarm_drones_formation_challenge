// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:action/DetectArucoMarkers.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__DETECT_ARUCO_MARKERS__STRUCT_H_
#define AS2_MSGS__ACTION__DETAIL__DETECT_ARUCO_MARKERS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'target_ids'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_Goal
{
  /// Request
  rosidl_runtime_c__uint16__Sequence target_ids;
} as2_msgs__action__DetectArucoMarkers_Goal;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_Goal.
typedef struct as2_msgs__action__DetectArucoMarkers_Goal__Sequence
{
  as2_msgs__action__DetectArucoMarkers_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_Result
{
  bool success;
} as2_msgs__action__DetectArucoMarkers_Result;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_Result.
typedef struct as2_msgs__action__DetectArucoMarkers_Result__Sequence
{
  as2_msgs__action__DetectArucoMarkers_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_Result__Sequence;


// Constants defined in the message

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_Feedback
{
  bool sucess;
} as2_msgs__action__DetectArucoMarkers_Feedback;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_Feedback.
typedef struct as2_msgs__action__DetectArucoMarkers_Feedback__Sequence
{
  as2_msgs__action__DetectArucoMarkers_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "as2_msgs/action/detail/detect_aruco_markers__struct.h"

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__DetectArucoMarkers_Goal goal;
} as2_msgs__action__DetectArucoMarkers_SendGoal_Request;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_SendGoal_Request.
typedef struct as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence
{
  as2_msgs__action__DetectArucoMarkers_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} as2_msgs__action__DetectArucoMarkers_SendGoal_Response;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_SendGoal_Response.
typedef struct as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence
{
  as2_msgs__action__DetectArucoMarkers_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} as2_msgs__action__DetectArucoMarkers_GetResult_Request;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_GetResult_Request.
typedef struct as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence
{
  as2_msgs__action__DetectArucoMarkers_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.h"

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_GetResult_Response
{
  int8_t status;
  as2_msgs__action__DetectArucoMarkers_Result result;
} as2_msgs__action__DetectArucoMarkers_GetResult_Response;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_GetResult_Response.
typedef struct as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence
{
  as2_msgs__action__DetectArucoMarkers_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/detect_aruco_markers__struct.h"

/// Struct defined in action/DetectArucoMarkers in the package as2_msgs.
typedef struct as2_msgs__action__DetectArucoMarkers_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__DetectArucoMarkers_Feedback feedback;
} as2_msgs__action__DetectArucoMarkers_FeedbackMessage;

// Struct for a sequence of as2_msgs__action__DetectArucoMarkers_FeedbackMessage.
typedef struct as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence
{
  as2_msgs__action__DetectArucoMarkers_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__ACTION__DETAIL__DETECT_ARUCO_MARKERS__STRUCT_H_
