// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/AlertEvent.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/alert_event__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `description`
#include "rosidl_runtime_c/string_functions.h"

bool
as2_msgs__msg__AlertEvent__init(as2_msgs__msg__AlertEvent * msg)
{
  if (!msg) {
    return false;
  }
  // alert
  // description
  if (!rosidl_runtime_c__String__init(&msg->description)) {
    as2_msgs__msg__AlertEvent__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__msg__AlertEvent__fini(as2_msgs__msg__AlertEvent * msg)
{
  if (!msg) {
    return;
  }
  // alert
  // description
  rosidl_runtime_c__String__fini(&msg->description);
}

bool
as2_msgs__msg__AlertEvent__are_equal(const as2_msgs__msg__AlertEvent * lhs, const as2_msgs__msg__AlertEvent * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // alert
  if (lhs->alert != rhs->alert) {
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->description), &(rhs->description)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__AlertEvent__copy(
  const as2_msgs__msg__AlertEvent * input,
  as2_msgs__msg__AlertEvent * output)
{
  if (!input || !output) {
    return false;
  }
  // alert
  output->alert = input->alert;
  // description
  if (!rosidl_runtime_c__String__copy(
      &(input->description), &(output->description)))
  {
    return false;
  }
  return true;
}

as2_msgs__msg__AlertEvent *
as2_msgs__msg__AlertEvent__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__AlertEvent * msg = (as2_msgs__msg__AlertEvent *)allocator.allocate(sizeof(as2_msgs__msg__AlertEvent), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__AlertEvent));
  bool success = as2_msgs__msg__AlertEvent__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__AlertEvent__destroy(as2_msgs__msg__AlertEvent * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__AlertEvent__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__AlertEvent__Sequence__init(as2_msgs__msg__AlertEvent__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__AlertEvent * data = NULL;

  if (size) {
    data = (as2_msgs__msg__AlertEvent *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__AlertEvent), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__AlertEvent__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__AlertEvent__fini(&data[i - 1]);
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
as2_msgs__msg__AlertEvent__Sequence__fini(as2_msgs__msg__AlertEvent__Sequence * array)
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
      as2_msgs__msg__AlertEvent__fini(&array->data[i]);
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

as2_msgs__msg__AlertEvent__Sequence *
as2_msgs__msg__AlertEvent__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__AlertEvent__Sequence * array = (as2_msgs__msg__AlertEvent__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__AlertEvent__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__AlertEvent__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__AlertEvent__Sequence__destroy(as2_msgs__msg__AlertEvent__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__AlertEvent__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__AlertEvent__Sequence__are_equal(const as2_msgs__msg__AlertEvent__Sequence * lhs, const as2_msgs__msg__AlertEvent__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__AlertEvent__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__AlertEvent__Sequence__copy(
  const as2_msgs__msg__AlertEvent__Sequence * input,
  as2_msgs__msg__AlertEvent__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__AlertEvent);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__AlertEvent * data =
      (as2_msgs__msg__AlertEvent *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__AlertEvent__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__AlertEvent__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__AlertEvent__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
