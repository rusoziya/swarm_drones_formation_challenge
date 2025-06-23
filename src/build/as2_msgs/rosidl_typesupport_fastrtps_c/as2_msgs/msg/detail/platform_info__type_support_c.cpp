// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from as2_msgs:msg/PlatformInfo.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/platform_info__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "as2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "as2_msgs/msg/detail/platform_info__struct.h"
#include "as2_msgs/msg/detail/platform_info__functions.h"
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

#include "as2_msgs/msg/detail/control_mode__functions.h"  // current_control_mode
#include "as2_msgs/msg/detail/platform_status__functions.h"  // status
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
size_t get_serialized_size_as2_msgs__msg__ControlMode(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_as2_msgs__msg__ControlMode(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, msg, ControlMode)();
size_t get_serialized_size_as2_msgs__msg__PlatformStatus(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_as2_msgs__msg__PlatformStatus(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, msg, PlatformStatus)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_as2_msgs
size_t get_serialized_size_std_msgs__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_as2_msgs
size_t max_serialized_size_std_msgs__msg__Header(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_as2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, Header)();


using _PlatformInfo__ros_msg_type = as2_msgs__msg__PlatformInfo;

static bool _PlatformInfo__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PlatformInfo__ros_msg_type * ros_message = static_cast<const _PlatformInfo__ros_msg_type *>(untyped_ros_message);
  // Field name: header
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Header
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->header, cdr))
    {
      return false;
    }
  }

  // Field name: connected
  {
    cdr << (ros_message->connected ? true : false);
  }

  // Field name: armed
  {
    cdr << (ros_message->armed ? true : false);
  }

  // Field name: offboard
  {
    cdr << (ros_message->offboard ? true : false);
  }

  // Field name: status
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, as2_msgs, msg, PlatformStatus
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->status, cdr))
    {
      return false;
    }
  }

  // Field name: current_control_mode
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, as2_msgs, msg, ControlMode
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->current_control_mode, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _PlatformInfo__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PlatformInfo__ros_msg_type * ros_message = static_cast<_PlatformInfo__ros_msg_type *>(untyped_ros_message);
  // Field name: header
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Header
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->header))
    {
      return false;
    }
  }

  // Field name: connected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->connected = tmp ? true : false;
  }

  // Field name: armed
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->armed = tmp ? true : false;
  }

  // Field name: offboard
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->offboard = tmp ? true : false;
  }

  // Field name: status
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, as2_msgs, msg, PlatformStatus
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->status))
    {
      return false;
    }
  }

  // Field name: current_control_mode
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, as2_msgs, msg, ControlMode
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->current_control_mode))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t get_serialized_size_as2_msgs__msg__PlatformInfo(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PlatformInfo__ros_msg_type * ros_message = static_cast<const _PlatformInfo__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name connected
  {
    size_t item_size = sizeof(ros_message->connected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name armed
  {
    size_t item_size = sizeof(ros_message->armed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name offboard
  {
    size_t item_size = sizeof(ros_message->offboard);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name status

  current_alignment += get_serialized_size_as2_msgs__msg__PlatformStatus(
    &(ros_message->status), current_alignment);
  // field.name current_control_mode

  current_alignment += get_serialized_size_as2_msgs__msg__ControlMode(
    &(ros_message->current_control_mode), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _PlatformInfo__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_as2_msgs__msg__PlatformInfo(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t max_serialized_size_as2_msgs__msg__PlatformInfo(
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

  // member: header
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__Header(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: connected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: armed
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: offboard
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: status
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_as2_msgs__msg__PlatformStatus(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: current_control_mode
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_as2_msgs__msg__ControlMode(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = as2_msgs__msg__PlatformInfo;
    is_plain =
      (
      offsetof(DataType, current_control_mode) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _PlatformInfo__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_as2_msgs__msg__PlatformInfo(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PlatformInfo = {
  "as2_msgs::msg",
  "PlatformInfo",
  _PlatformInfo__cdr_serialize,
  _PlatformInfo__cdr_deserialize,
  _PlatformInfo__get_serialized_size,
  _PlatformInfo__max_serialized_size
};

static rosidl_message_type_support_t _PlatformInfo__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PlatformInfo,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, msg, PlatformInfo)() {
  return &_PlatformInfo__type_support;
}

#if defined(__cplusplus)
}
#endif
