// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:action/Takeoff.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__ACTION__DETAIL__TAKEOFF__STRUCT_H_
#define AS2_MSGS__ACTION__DETAIL__TAKEOFF__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_Goal
{
  /// Request
  /// Takeoff height (m)
  float takeoff_height;
  /// Takeoff speed (m/s)
  float takeoff_speed;
} as2_msgs__action__Takeoff_Goal;

// Struct for a sequence of as2_msgs__action__Takeoff_Goal.
typedef struct as2_msgs__action__Takeoff_Goal__Sequence
{
  as2_msgs__action__Takeoff_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_Goal__Sequence;


// Constants defined in the message

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_Result
{
  /// false if failed to takeoff
  bool takeoff_success;
} as2_msgs__action__Takeoff_Result;

// Struct for a sequence of as2_msgs__action__Takeoff_Result.
typedef struct as2_msgs__action__Takeoff_Result__Sequence
{
  as2_msgs__action__Takeoff_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_Result__Sequence;


// Constants defined in the message

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_Feedback
{
  /// actual speed (m/s)
  float actual_takeoff_speed;
  /// actual height (m)
  float actual_takeoff_height;
} as2_msgs__action__Takeoff_Feedback;

// Struct for a sequence of as2_msgs__action__Takeoff_Feedback.
typedef struct as2_msgs__action__Takeoff_Feedback__Sequence
{
  as2_msgs__action__Takeoff_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "as2_msgs/action/detail/takeoff__struct.h"

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__Takeoff_Goal goal;
} as2_msgs__action__Takeoff_SendGoal_Request;

// Struct for a sequence of as2_msgs__action__Takeoff_SendGoal_Request.
typedef struct as2_msgs__action__Takeoff_SendGoal_Request__Sequence
{
  as2_msgs__action__Takeoff_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} as2_msgs__action__Takeoff_SendGoal_Response;

// Struct for a sequence of as2_msgs__action__Takeoff_SendGoal_Response.
typedef struct as2_msgs__action__Takeoff_SendGoal_Response__Sequence
{
  as2_msgs__action__Takeoff_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} as2_msgs__action__Takeoff_GetResult_Request;

// Struct for a sequence of as2_msgs__action__Takeoff_GetResult_Request.
typedef struct as2_msgs__action__Takeoff_GetResult_Request__Sequence
{
  as2_msgs__action__Takeoff_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "as2_msgs/action/detail/takeoff__struct.h"

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_GetResult_Response
{
  int8_t status;
  as2_msgs__action__Takeoff_Result result;
} as2_msgs__action__Takeoff_GetResult_Response;

// Struct for a sequence of as2_msgs__action__Takeoff_GetResult_Response.
typedef struct as2_msgs__action__Takeoff_GetResult_Response__Sequence
{
  as2_msgs__action__Takeoff_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "as2_msgs/action/detail/takeoff__struct.h"

/// Struct defined in action/Takeoff in the package as2_msgs.
typedef struct as2_msgs__action__Takeoff_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  as2_msgs__action__Takeoff_Feedback feedback;
} as2_msgs__action__Takeoff_FeedbackMessage;

// Struct for a sequence of as2_msgs__action__Takeoff_FeedbackMessage.
typedef struct as2_msgs__action__Takeoff_FeedbackMessage__Sequence
{
  as2_msgs__action__Takeoff_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__action__Takeoff_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__ACTION__DETAIL__TAKEOFF__STRUCT_H_
