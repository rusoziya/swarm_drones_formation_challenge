// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from as2_msgs:msg/PolygonList.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "as2_msgs/msg/detail/polygon_list__rosidl_typesupport_introspection_c.h"
#include "as2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "as2_msgs/msg/detail/polygon_list__functions.h"
#include "as2_msgs/msg/detail/polygon_list__struct.h"


// Include directives for member types
// Member `polygons`
#include "geometry_msgs/msg/polygon_stamped.h"
// Member `polygons`
#include "geometry_msgs/msg/detail/polygon_stamped__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  as2_msgs__msg__PolygonList__init(message_memory);
}

void as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_fini_function(void * message_memory)
{
  as2_msgs__msg__PolygonList__fini(message_memory);
}

size_t as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__size_function__PolygonList__polygons(
  const void * untyped_member)
{
  const geometry_msgs__msg__PolygonStamped__Sequence * member =
    (const geometry_msgs__msg__PolygonStamped__Sequence *)(untyped_member);
  return member->size;
}

const void * as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__get_const_function__PolygonList__polygons(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__PolygonStamped__Sequence * member =
    (const geometry_msgs__msg__PolygonStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void * as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__get_function__PolygonList__polygons(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__PolygonStamped__Sequence * member =
    (geometry_msgs__msg__PolygonStamped__Sequence *)(untyped_member);
  return &member->data[index];
}

void as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__fetch_function__PolygonList__polygons(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__PolygonStamped * item =
    ((const geometry_msgs__msg__PolygonStamped *)
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__get_const_function__PolygonList__polygons(untyped_member, index));
  geometry_msgs__msg__PolygonStamped * value =
    (geometry_msgs__msg__PolygonStamped *)(untyped_value);
  *value = *item;
}

void as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__assign_function__PolygonList__polygons(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__PolygonStamped * item =
    ((geometry_msgs__msg__PolygonStamped *)
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__get_function__PolygonList__polygons(untyped_member, index));
  const geometry_msgs__msg__PolygonStamped * value =
    (const geometry_msgs__msg__PolygonStamped *)(untyped_value);
  *item = *value;
}

bool as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__resize_function__PolygonList__polygons(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__PolygonStamped__Sequence * member =
    (geometry_msgs__msg__PolygonStamped__Sequence *)(untyped_member);
  geometry_msgs__msg__PolygonStamped__Sequence__fini(member);
  return geometry_msgs__msg__PolygonStamped__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_member_array[1] = {
  {
    "polygons",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(as2_msgs__msg__PolygonList, polygons),  // bytes offset in struct
    NULL,  // default value
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__size_function__PolygonList__polygons,  // size() function pointer
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__get_const_function__PolygonList__polygons,  // get_const(index) function pointer
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__get_function__PolygonList__polygons,  // get(index) function pointer
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__fetch_function__PolygonList__polygons,  // fetch(index, &value) function pointer
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__assign_function__PolygonList__polygons,  // assign(index, value) function pointer
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__resize_function__PolygonList__polygons  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_members = {
  "as2_msgs__msg",  // message namespace
  "PolygonList",  // message name
  1,  // number of fields
  sizeof(as2_msgs__msg__PolygonList),
  as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_member_array,  // message members
  as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_init_function,  // function to initialize message memory (memory has to be allocated)
  as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_type_support_handle = {
  0,
  &as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_as2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, as2_msgs, msg, PolygonList)() {
  as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PolygonStamped)();
  if (!as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_type_support_handle.typesupport_identifier) {
    as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &as2_msgs__msg__PolygonList__rosidl_typesupport_introspection_c__PolygonList_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
