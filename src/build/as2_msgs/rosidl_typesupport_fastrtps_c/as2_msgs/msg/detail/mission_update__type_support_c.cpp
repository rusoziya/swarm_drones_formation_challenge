// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from as2_msgs:msg/MissionUpdate.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/mission_update__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "as2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "as2_msgs/msg/detail/mission_update__struct.h"
#include "as2_msgs/msg/detail/mission_update__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/string.h"  // drone_id, mission
#include "rosidl_runtime_c/string_functions.h"  // drone_id, mission

// forward declare type support functions


using _MissionUpdate__ros_msg_type = as2_msgs__msg__MissionUpdate;

static bool _MissionUpdate__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _MissionUpdate__ros_msg_type * ros_message = static_cast<const _MissionUpdate__ros_msg_type *>(untyped_ros_message);
  // Field name: drone_id
  {
    const rosidl_runtime_c__String * str = &ros_message->drone_id;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: mission_id
  {
    cdr << ros_message->mission_id;
  }

  // Field name: item_id
  {
    cdr << ros_message->item_id;
  }

  // Field name: action
  {
    cdr << ros_message->action;
  }

  // Field name: mission
  {
    const rosidl_runtime_c__String * str = &ros_message->mission;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  return true;
}

static bool _MissionUpdate__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _MissionUpdate__ros_msg_type * ros_message = static_cast<_MissionUpdate__ros_msg_type *>(untyped_ros_message);
  // Field name: drone_id
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->drone_id.data) {
      rosidl_runtime_c__String__init(&ros_message->drone_id);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->drone_id,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'drone_id'\n");
      return false;
    }
  }

  // Field name: mission_id
  {
    cdr >> ros_message->mission_id;
  }

  // Field name: item_id
  {
    cdr >> ros_message->item_id;
  }

  // Field name: action
  {
    cdr >> ros_message->action;
  }

  // Field name: mission
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->mission.data) {
      rosidl_runtime_c__String__init(&ros_message->mission);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->mission,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'mission'\n");
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t get_serialized_size_as2_msgs__msg__MissionUpdate(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _MissionUpdate__ros_msg_type * ros_message = static_cast<const _MissionUpdate__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name drone_id
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->drone_id.size + 1);
  // field.name mission_id
  {
    size_t item_size = sizeof(ros_message->mission_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name item_id
  {
    size_t item_size = sizeof(ros_message->item_id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name action
  {
    size_t item_size = sizeof(ros_message->action);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name mission
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->mission.size + 1);

  return current_alignment - initial_alignment;
}

static uint32_t _MissionUpdate__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_as2_msgs__msg__MissionUpdate(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t max_serialized_size_as2_msgs__msg__MissionUpdate(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: drone_id
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: mission_id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: item_id
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: action
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: mission
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = as2_msgs__msg__MissionUpdate;
    is_plain =
      (
      offsetof(DataType, mission) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _MissionUpdate__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_as2_msgs__msg__MissionUpdate(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_MissionUpdate = {
  "as2_msgs::msg",
  "MissionUpdate",
  _MissionUpdate__cdr_serialize,
  _MissionUpdate__cdr_deserialize,
  _MissionUpdate__get_serialized_size,
  _MissionUpdate__max_serialized_size
};

static rosidl_message_type_support_t _MissionUpdate__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_MissionUpdate,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, msg, MissionUpdate)() {
  return &_MissionUpdate__type_support;
}

#if defined(__cplusplus)
}
#endif
