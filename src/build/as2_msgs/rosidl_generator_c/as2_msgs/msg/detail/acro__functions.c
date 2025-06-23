// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/Acro.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/acro__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `angular_rates`
// Member `thrust`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
as2_msgs__msg__Acro__init(as2_msgs__msg__Acro * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    as2_msgs__msg__Acro__fini(msg);
    return false;
  }
  // angular_rates
  if (!geometry_msgs__msg__Vector3__init(&msg->angular_rates)) {
    as2_msgs__msg__Acro__fini(msg);
    return false;
  }
  // thrust
  if (!geometry_msgs__msg__Vector3__init(&msg->thrust)) {
    as2_msgs__msg__Acro__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__msg__Acro__fini(as2_msgs__msg__Acro * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // angular_rates
  geometry_msgs__msg__Vector3__fini(&msg->angular_rates);
  // thrust
  geometry_msgs__msg__Vector3__fini(&msg->thrust);
}

bool
as2_msgs__msg__Acro__are_equal(const as2_msgs__msg__Acro * lhs, const as2_msgs__msg__Acro * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // angular_rates
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->angular_rates), &(rhs->angular_rates)))
  {
    return false;
  }
  // thrust
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->thrust), &(rhs->thrust)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__Acro__copy(
  const as2_msgs__msg__Acro * input,
  as2_msgs__msg__Acro * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // angular_rates
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->angular_rates), &(output->angular_rates)))
  {
    return false;
  }
  // thrust
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->thrust), &(output->thrust)))
  {
    return false;
  }
  return true;
}

as2_msgs__msg__Acro *
as2_msgs__msg__Acro__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__Acro * msg = (as2_msgs__msg__Acro *)allocator.allocate(sizeof(as2_msgs__msg__Acro), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__Acro));
  bool success = as2_msgs__msg__Acro__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__Acro__destroy(as2_msgs__msg__Acro * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__Acro__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__Acro__Sequence__init(as2_msgs__msg__Acro__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__Acro * data = NULL;

  if (size) {
    data = (as2_msgs__msg__Acro *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__Acro), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__Acro__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__Acro__fini(&data[i - 1]);
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
as2_msgs__msg__Acro__Sequence__fini(as2_msgs__msg__Acro__Sequence * array)
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
      as2_msgs__msg__Acro__fini(&array->data[i]);
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

as2_msgs__msg__Acro__Sequence *
as2_msgs__msg__Acro__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__Acro__Sequence * array = (as2_msgs__msg__Acro__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__Acro__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__Acro__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__Acro__Sequence__destroy(as2_msgs__msg__Acro__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__Acro__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__Acro__Sequence__are_equal(const as2_msgs__msg__Acro__Sequence * lhs, const as2_msgs__msg__Acro__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__Acro__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__Acro__Sequence__copy(
  const as2_msgs__msg__Acro__Sequence * input,
  as2_msgs__msg__Acro__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__Acro);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__Acro * data =
      (as2_msgs__msg__Acro *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__Acro__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__Acro__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__Acro__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
