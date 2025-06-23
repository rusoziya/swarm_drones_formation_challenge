// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from as2_msgs:msg/MissionUpdate.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__STRUCT_H_
#define AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'EXECUTE'.
/**
  * Execute a mission in the interpreter
 */
enum
{
  as2_msgs__msg__MissionUpdate__EXECUTE = 0
};

/// Constant 'LOAD'.
/**
  * Load a mission to the interpreter
 */
enum
{
  as2_msgs__msg__MissionUpdate__LOAD = 1
};

/// Constant 'START'.
/**
  * Start the execution of a mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__START = 2
};

/// Constant 'PAUSE'.
/**
  * Pause the execution of a mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__PAUSE = 3
};

/// Constant 'RESUME'.
/**
  * Resume the execution of a mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__RESUME = 4
};

/// Constant 'STOP'.
/**
  * Stop the execution of a mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__STOP = 5
};

/// Constant 'JUMP_TO'.
/**
  * Jump to a specific item in the mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__JUMP_TO = 6
};

/// Constant 'REPEAT'.
/**
  * Repeat the execution of a mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__REPEAT = 7
};

/// Constant 'INSERT'.
/**
  * Insert an item in the mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__INSERT = 8
};

/// Constant 'MODIFY'.
/**
  * Modify an item in the mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__MODIFY = 9
};

/// Constant 'REMOVE'.
/**
  * Remove an item in the mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__REMOVE = 10
};

/// Constant 'RESET'.
/**
  * Reset the interpreter
 */
enum
{
  as2_msgs__msg__MissionUpdate__RESET = 11
};

/// Constant 'ABORT'.
/**
  * Abort the execution of a mission
 */
enum
{
  as2_msgs__msg__MissionUpdate__ABORT = 12
};

// Include directives for member types
// Member 'drone_id'
// Member 'mission'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/MissionUpdate in the package as2_msgs.
/**
  * Message that sends a mission to the interpreter
 */
typedef struct as2_msgs__msg__MissionUpdate
{
  /// ID of the drone that will execute the mission
  rosidl_runtime_c__String drone_id;
  /// ID of the mission to be executed
  int32_t mission_id;
  /// ID of the item to be executed
  int32_t item_id;
  /// Action to be performed in the interpreter
  uint8_t action;
  /// JSON formatted mission to be executed
  rosidl_runtime_c__String mission;
} as2_msgs__msg__MissionUpdate;

// Struct for a sequence of as2_msgs__msg__MissionUpdate.
typedef struct as2_msgs__msg__MissionUpdate__Sequence
{
  as2_msgs__msg__MissionUpdate * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} as2_msgs__msg__MissionUpdate__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__MISSION_UPDATE__STRUCT_H_
