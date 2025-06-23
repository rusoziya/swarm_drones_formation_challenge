// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from as2_msgs:msg/PoseStampedWithIDArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "as2_msgs/msg/detail/pose_stamped_with_id_array__rosidl_typesupport_introspection_c.h"
#include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "as2_msgs/msg/detail/pose_stamped_with_id_array__functions.h"
#include "as2_msgs/msg/detail/pose_stamped_with_id_array__struct.h"


// Include directives for member types
// Member `poses`
#include "as2_msgs/msg/pose_stamped_with_id.h"
// Member `poses`
#include "as2_msgs/msg/detail/pose_stamped_with_id__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  as2_msgs__msg__PoseStampedWithIDArray__init(message_memory);
}

void as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_fini_function(void * message_memory)
{
  as2_msgs__msg__PoseStampedWithIDArray__fini(message_memory);
}

size_t as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__size_function__PoseStampedWithIDArray__poses(
  const void * untyped_member)
{
  const as2_msgs__msg__PoseStampedWithID__Sequence * member =
    (const as2_msgs__msg__PoseStampedWithID__Sequence *)(untyped_member);
  return member->size;
}

const void * as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__get_const_function__PoseStampedWithIDArray__poses(
  const void * untyped_member, size_t index)
{
  const as2_msgs__msg__PoseStampedWithID__Sequence * member =
    (const as2_msgs__msg__PoseStampedWithID__Sequence *)(untyped_member);
  return &member->data[index];
}

void * as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__get_function__PoseStampedWithIDArray__poses(
  void * untyped_member, size_t index)
{
  as2_msgs__msg__PoseStampedWithID__Sequence * member =
    (as2_msgs__msg__PoseStampedWithID__Sequence *)(untyped_member);
  return &member->data[index];
}

void as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__fetch_function__PoseStampedWithIDArray__poses(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const as2_msgs__msg__PoseStampedWithID * item =
    ((const as2_msgs__msg__PoseStampedWithID *)
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__get_const_function__PoseStampedWithIDArray__poses(untyped_member, index));
  as2_msgs__msg__PoseStampedWithID * value =
    (as2_msgs__msg__PoseStampedWithID *)(untyped_value);
  *value = *item;
}

void as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__assign_function__PoseStampedWithIDArray__poses(
  void * untyped_member, size_t index, const void * untyped_value)
{
  as2_msgs__msg__PoseStampedWithID * item =
    ((as2_msgs__msg__PoseStampedWithID *)
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__get_function__PoseStampedWithIDArray__poses(untyped_member, index));
  const as2_msgs__msg__PoseStampedWithID * value =
    (const as2_msgs__msg__PoseStampedWithID *)(untyped_value);
  *item = *value;
}

bool as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__resize_function__PoseStampedWithIDArray__poses(
  void * untyped_member, size_t size)
{
  as2_msgs__msg__PoseStampedWithID__Sequence * member =
    (as2_msgs__msg__PoseStampedWithID__Sequence *)(untyped_member);
  as2_msgs__msg__PoseStampedWithID__Sequence__fini(member);
  return as2_msgs__msg__PoseStampedWithID__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_member_array[1] = {
  {
    "poses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__PoseStampedWithIDArray, poses),  // bytes offset in struct
    NULL,  // default value
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__size_function__PoseStampedWithIDArray__poses,  // size() function pointer
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__get_const_function__PoseStampedWithIDArray__poses,  // get_const(index) function pointer
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__get_function__PoseStampedWithIDArray__poses,  // get(index) function pointer
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__fetch_function__PoseStampedWithIDArray__poses,  // fetch(index, &value) function pointer
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__assign_function__PoseStampedWithIDArray__poses,  // assign(index, value) function pointer
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__resize_function__PoseStampedWithIDArray__poses  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_members = {
  "as2_msgs__msg",  // message namespace
  "PoseStampedWithIDArray",  // message name
  1,  // number of fields
  sizeof(as2_msgs__msg__PoseStampedWithIDArray),
  as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_member_array,  // message members
  as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_init_function,  // function to initialize message memory (memory has to be allocated)
  as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_type_support_handle = {
  0,
  &as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, msg, PoseStampedWithIDArray)() {
  as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, msg, PoseStampedWithID)();
  if (!as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_type_support_handle.typesupport_identifier) {
    as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &as2_msgs__msg__PoseStampedWithIDArray__rosidl_typesupport_introspection_c__PoseStampedWithIDArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
