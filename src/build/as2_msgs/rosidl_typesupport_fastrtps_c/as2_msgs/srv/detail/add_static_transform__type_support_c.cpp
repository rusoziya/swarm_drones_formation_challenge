// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from as2_msgs:srv/AddStaticTransform.idl
// generated code does not contain a copyright notice
#include "as2_msgs/srv/detail/add_static_transform__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "as2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "as2_msgs/srv/detail/add_static_transform__struct.h"
#include "as2_msgs/srv/detail/add_static_transform__functions.h"
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

#include "geometry_msgs/msg/detail/transform__functions.h"  // transform
#include "rosidl_runtime_c/string.h"  // child_frame_id, frame_id
#include "rosidl_runtime_c/string_functions.h"  // child_frame_id, frame_id

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_as2_msgs
size_t get_serialized_size_geometry_msgs__msg__Transform(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_as2_msgs
size_t max_serialized_size_geometry_msgs__msg__Transform(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_as2_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, geometry_msgs, msg, Transform)();


using _AddStaticTransform_Request__ros_msg_type = as2_msgs__srv__AddStaticTransform_Request;

static bool _AddStaticTransform_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _AddStaticTransform_Request__ros_msg_type * ros_message = static_cast<const _AddStaticTransform_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: frame_id
  {
    const rosidl_runtime_c__String * str = &ros_message->frame_id;
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

  // Field name: child_frame_id
  {
    const rosidl_runtime_c__String * str = &ros_message->child_frame_id;
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

  // Field name: transform
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, geometry_msgs, msg, Transform
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->transform, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _AddStaticTransform_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _AddStaticTransform_Request__ros_msg_type * ros_message = static_cast<_AddStaticTransform_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: frame_id
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->frame_id.data) {
      rosidl_runtime_c__String__init(&ros_message->frame_id);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->frame_id,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'frame_id'\n");
      return false;
    }
  }

  // Field name: child_frame_id
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->child_frame_id.data) {
      rosidl_runtime_c__String__init(&ros_message->child_frame_id);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->child_frame_id,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'child_frame_id'\n");
      return false;
    }
  }

  // Field name: transform
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, geometry_msgs, msg, Transform
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->transform))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t get_serialized_size_as2_msgs__srv__AddStaticTransform_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _AddStaticTransform_Request__ros_msg_type * ros_message = static_cast<const _AddStaticTransform_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name frame_id
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->frame_id.size + 1);
  // field.name child_frame_id
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->child_frame_id.size + 1);
  // field.name transform

  current_alignment += get_serialized_size_geometry_msgs__msg__Transform(
    &(ros_message->transform), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _AddStaticTransform_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_as2_msgs__srv__AddStaticTransform_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t max_serialized_size_as2_msgs__srv__AddStaticTransform_Request(
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

  // member: frame_id
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
  // member: child_frame_id
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
  // member: transform
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_geometry_msgs__msg__Transform(
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
    using DataType = as2_msgs__srv__AddStaticTransform_Request;
    is_plain =
      (
      offsetof(DataType, transform) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _AddStaticTransform_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_as2_msgs__srv__AddStaticTransform_Request(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_AddStaticTransform_Request = {
  "as2_msgs::srv",
  "AddStaticTransform_Request",
  _AddStaticTransform_Request__cdr_serialize,
  _AddStaticTransform_Request__cdr_deserialize,
  _AddStaticTransform_Request__get_serialized_size,
  _AddStaticTransform_Request__max_serialized_size
};

static rosidl_message_type_support_t _AddStaticTransform_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_AddStaticTransform_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, srv, AddStaticTransform_Request)() {
  return &_AddStaticTransform_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "as2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "as2_msgs/srv/detail/add_static_transform__struct.h"
// already included above
// #include "as2_msgs/srv/detail/add_static_transform__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

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


// forward declare type support functions


using _AddStaticTransform_Response__ros_msg_type = as2_msgs__srv__AddStaticTransform_Response;

static bool _AddStaticTransform_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _AddStaticTransform_Response__ros_msg_type * ros_message = static_cast<const _AddStaticTransform_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: success
  {
    cdr << (ros_message->success ? true : false);
  }

  return true;
}

static bool _AddStaticTransform_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _AddStaticTransform_Response__ros_msg_type * ros_message = static_cast<_AddStaticTransform_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: success
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->success = tmp ? true : false;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t get_serialized_size_as2_msgs__srv__AddStaticTransform_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _AddStaticTransform_Response__ros_msg_type * ros_message = static_cast<const _AddStaticTransform_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name success
  {
    size_t item_size = sizeof(ros_message->success);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _AddStaticTransform_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_as2_msgs__srv__AddStaticTransform_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_as2_msgs
size_t max_serialized_size_as2_msgs__srv__AddStaticTransform_Response(
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

  // member: success
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = as2_msgs__srv__AddStaticTransform_Response;
    is_plain =
      (
      offsetof(DataType, success) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _AddStaticTransform_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_as2_msgs__srv__AddStaticTransform_Response(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_AddStaticTransform_Response = {
  "as2_msgs::srv",
  "AddStaticTransform_Response",
  _AddStaticTransform_Response__cdr_serialize,
  _AddStaticTransform_Response__cdr_deserialize,
  _AddStaticTransform_Response__get_serialized_size,
  _AddStaticTransform_Response__max_serialized_size
};

static rosidl_message_type_support_t _AddStaticTransform_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_AddStaticTransform_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, srv, AddStaticTransform_Response)() {
  return &_AddStaticTransform_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "as2_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "as2_msgs/srv/add_static_transform.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t AddStaticTransform__callbacks = {
  "as2_msgs::srv",
  "AddStaticTransform",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, srv, AddStaticTransform_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, srv, AddStaticTransform_Response)(),
};

static rosidl_service_type_support_t AddStaticTransform__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &AddStaticTransform__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, as2_msgs, srv, AddStaticTransform)() {
  return &AddStaticTransform__handle;
}

#if defined(__cplusplus)
}
#endif
