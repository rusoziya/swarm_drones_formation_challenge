// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/MissionUpdate.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/mission_update__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `drone_id`
// Member `mission`
#include "rosidl_runtime_c/string_functions.h"

bool
as2_msgs__msg__MissionUpdate__init(as2_msgs__msg__MissionUpdate * msg)
{
  if (!msg) {
    return false;
  }
  // drone_id
  if (!rosidl_runtime_c__String__init(&msg->drone_id)) {
    as2_msgs__msg__MissionUpdate__fini(msg);
    return false;
  }
  // mission_id
  // item_id
  // action
  // mission
  if (!rosidl_runtime_c__String__init(&msg->mission)) {
    as2_msgs__msg__MissionUpdate__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__msg__MissionUpdate__fini(as2_msgs__msg__MissionUpdate * msg)
{
  if (!msg) {
    return;
  }
  // drone_id
  rosidl_runtime_c__String__fini(&msg->drone_id);
  // mission_id
  // item_id
  // action
  // mission
  rosidl_runtime_c__String__fini(&msg->mission);
}

bool
as2_msgs__msg__MissionUpdate__are_equal(const as2_msgs__msg__MissionUpdate * lhs, const as2_msgs__msg__MissionUpdate * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // drone_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->drone_id), &(rhs->drone_id)))
  {
    return false;
  }
  // mission_id
  if (lhs->mission_id != rhs->mission_id) {
    return false;
  }
  // item_id
  if (lhs->item_id != rhs->item_id) {
    return false;
  }
  // action
  if (lhs->action != rhs->action) {
    return false;
  }
  // mission
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->mission), &(rhs->mission)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__MissionUpdate__copy(
  const as2_msgs__msg__MissionUpdate * input,
  as2_msgs__msg__MissionUpdate * output)
{
  if (!input || !output) {
    return false;
  }
  // drone_id
  if (!rosidl_runtime_c__String__copy(
      &(input->drone_id), &(output->drone_id)))
  {
    return false;
  }
  // mission_id
  output->mission_id = input->mission_id;
  // item_id
  output->item_id = input->item_id;
  // action
  output->action = input->action;
  // mission
  if (!rosidl_runtime_c__String__copy(
      &(input->mission), &(output->mission)))
  {
    return false;
  }
  return true;
}

as2_msgs__msg__MissionUpdate *
as2_msgs__msg__MissionUpdate__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__MissionUpdate * msg = (as2_msgs__msg__MissionUpdate *)allocator.allocate(sizeof(as2_msgs__msg__MissionUpdate), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__MissionUpdate));
  bool success = as2_msgs__msg__MissionUpdate__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__MissionUpdate__destroy(as2_msgs__msg__MissionUpdate * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__MissionUpdate__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__MissionUpdate__Sequence__init(as2_msgs__msg__MissionUpdate__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__MissionUpdate * data = NULL;

  if (size) {
    data = (as2_msgs__msg__MissionUpdate *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__MissionUpdate), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__MissionUpdate__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__MissionUpdate__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
as2_msgs__msg__MissionUpdate__Sequence__fini(as2_msgs__msg__MissionUpdate__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      as2_msgs__msg__MissionUpdate__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

as2_msgs__msg__MissionUpdate__Sequence *
as2_msgs__msg__MissionUpdate__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__MissionUpdate__Sequence * array = (as2_msgs__msg__MissionUpdate__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__MissionUpdate__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__MissionUpdate__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__MissionUpdate__Sequence__destroy(as2_msgs__msg__MissionUpdate__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__MissionUpdate__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__MissionUpdate__Sequence__are_equal(const as2_msgs__msg__MissionUpdate__Sequence * lhs, const as2_msgs__msg__MissionUpdate__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__MissionUpdate__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__MissionUpdate__Sequence__copy(
  const as2_msgs__msg__MissionUpdate__Sequence * input,
  as2_msgs__msg__MissionUpdate__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__MissionUpdate);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__MissionUpdate * data =
      (as2_msgs__msg__MissionUpdate *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__MissionUpdate__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__MissionUpdate__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__MissionUpdate__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
