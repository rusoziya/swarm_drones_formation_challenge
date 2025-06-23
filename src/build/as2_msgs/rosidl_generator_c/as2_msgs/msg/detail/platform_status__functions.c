// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/PlatformStatus.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/platform_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
as2_msgs__msg__PlatformStatus__init(as2_msgs__msg__PlatformStatus * msg)
{
  if (!msg) {
    return false;
  }
  // state
  return true;
}

void
as2_msgs__msg__PlatformStatus__fini(as2_msgs__msg__PlatformStatus * msg)
{
  if (!msg) {
    return;
  }
  // state
}

bool
as2_msgs__msg__PlatformStatus__are_equal(const as2_msgs__msg__PlatformStatus * lhs, const as2_msgs__msg__PlatformStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // state
  if (lhs->state != rhs->state) {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__PlatformStatus__copy(
  const as2_msgs__msg__PlatformStatus * input,
  as2_msgs__msg__PlatformStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // state
  output->state = input->state;
  return true;
}

as2_msgs__msg__PlatformStatus *
as2_msgs__msg__PlatformStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__PlatformStatus * msg = (as2_msgs__msg__PlatformStatus *)allocator.allocate(sizeof(as2_msgs__msg__PlatformStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__PlatformStatus));
  bool success = as2_msgs__msg__PlatformStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__PlatformStatus__destroy(as2_msgs__msg__PlatformStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__PlatformStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__PlatformStatus__Sequence__init(as2_msgs__msg__PlatformStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__PlatformStatus * data = NULL;

  if (size) {
    data = (as2_msgs__msg__PlatformStatus *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__PlatformStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__PlatformStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__PlatformStatus__fini(&data[i - 1]);
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
as2_msgs__msg__PlatformStatus__Sequence__fini(as2_msgs__msg__PlatformStatus__Sequence * array)
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
      as2_msgs__msg__PlatformStatus__fini(&array->data[i]);
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

as2_msgs__msg__PlatformStatus__Sequence *
as2_msgs__msg__PlatformStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__PlatformStatus__Sequence * array = (as2_msgs__msg__PlatformStatus__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__PlatformStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__PlatformStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__PlatformStatus__Sequence__destroy(as2_msgs__msg__PlatformStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__PlatformStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__PlatformStatus__Sequence__are_equal(const as2_msgs__msg__PlatformStatus__Sequence * lhs, const as2_msgs__msg__PlatformStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__PlatformStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__PlatformStatus__Sequence__copy(
  const as2_msgs__msg__PlatformStatus__Sequence * input,
  as2_msgs__msg__PlatformStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__PlatformStatus);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__PlatformStatus * data =
      (as2_msgs__msg__PlatformStatus *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__PlatformStatus__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__PlatformStatus__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__PlatformStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
