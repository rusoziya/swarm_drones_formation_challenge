// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/YawMode.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/yaw_mode__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
as2_msgs__msg__YawMode__init(as2_msgs__msg__YawMode * msg)
{
  if (!msg) {
    return false;
  }
  // mode
  // angle
  return true;
}

void
as2_msgs__msg__YawMode__fini(as2_msgs__msg__YawMode * msg)
{
  if (!msg) {
    return;
  }
  // mode
  // angle
}

bool
as2_msgs__msg__YawMode__are_equal(const as2_msgs__msg__YawMode * lhs, const as2_msgs__msg__YawMode * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  // angle
  if (lhs->angle != rhs->angle) {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__YawMode__copy(
  const as2_msgs__msg__YawMode * input,
  as2_msgs__msg__YawMode * output)
{
  if (!input || !output) {
    return false;
  }
  // mode
  output->mode = input->mode;
  // angle
  output->angle = input->angle;
  return true;
}

as2_msgs__msg__YawMode *
as2_msgs__msg__YawMode__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__YawMode * msg = (as2_msgs__msg__YawMode *)allocator.allocate(sizeof(as2_msgs__msg__YawMode), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__YawMode));
  bool success = as2_msgs__msg__YawMode__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__YawMode__destroy(as2_msgs__msg__YawMode * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__YawMode__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__YawMode__Sequence__init(as2_msgs__msg__YawMode__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__YawMode * data = NULL;

  if (size) {
    data = (as2_msgs__msg__YawMode *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__YawMode), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__YawMode__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__YawMode__fini(&data[i - 1]);
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
as2_msgs__msg__YawMode__Sequence__fini(as2_msgs__msg__YawMode__Sequence * array)
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
      as2_msgs__msg__YawMode__fini(&array->data[i]);
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

as2_msgs__msg__YawMode__Sequence *
as2_msgs__msg__YawMode__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__YawMode__Sequence * array = (as2_msgs__msg__YawMode__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__YawMode__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__YawMode__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__YawMode__Sequence__destroy(as2_msgs__msg__YawMode__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__YawMode__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__YawMode__Sequence__are_equal(const as2_msgs__msg__YawMode__Sequence * lhs, const as2_msgs__msg__YawMode__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__YawMode__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__YawMode__Sequence__copy(
  const as2_msgs__msg__YawMode__Sequence * input,
  as2_msgs__msg__YawMode__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__YawMode);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__YawMode * data =
      (as2_msgs__msg__YawMode *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__YawMode__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__YawMode__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__YawMode__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
