// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/ControlMode.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/control_mode__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
as2_msgs__msg__ControlMode__init(as2_msgs__msg__ControlMode * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    as2_msgs__msg__ControlMode__fini(msg);
    return false;
  }
  // yaw_mode
  // control_mode
  // reference_frame
  return true;
}

void
as2_msgs__msg__ControlMode__fini(as2_msgs__msg__ControlMode * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // yaw_mode
  // control_mode
  // reference_frame
}

bool
as2_msgs__msg__ControlMode__are_equal(const as2_msgs__msg__ControlMode * lhs, const as2_msgs__msg__ControlMode * rhs)
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
  // yaw_mode
  if (lhs->yaw_mode != rhs->yaw_mode) {
    return false;
  }
  // control_mode
  if (lhs->control_mode != rhs->control_mode) {
    return false;
  }
  // reference_frame
  if (lhs->reference_frame != rhs->reference_frame) {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__ControlMode__copy(
  const as2_msgs__msg__ControlMode * input,
  as2_msgs__msg__ControlMode * output)
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
  // yaw_mode
  output->yaw_mode = input->yaw_mode;
  // control_mode
  output->control_mode = input->control_mode;
  // reference_frame
  output->reference_frame = input->reference_frame;
  return true;
}

as2_msgs__msg__ControlMode *
as2_msgs__msg__ControlMode__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__ControlMode * msg = (as2_msgs__msg__ControlMode *)allocator.allocate(sizeof(as2_msgs__msg__ControlMode), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__ControlMode));
  bool success = as2_msgs__msg__ControlMode__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__ControlMode__destroy(as2_msgs__msg__ControlMode * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__ControlMode__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__ControlMode__Sequence__init(as2_msgs__msg__ControlMode__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__ControlMode * data = NULL;

  if (size) {
    data = (as2_msgs__msg__ControlMode *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__ControlMode), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__ControlMode__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__ControlMode__fini(&data[i - 1]);
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
as2_msgs__msg__ControlMode__Sequence__fini(as2_msgs__msg__ControlMode__Sequence * array)
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
      as2_msgs__msg__ControlMode__fini(&array->data[i]);
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

as2_msgs__msg__ControlMode__Sequence *
as2_msgs__msg__ControlMode__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__ControlMode__Sequence * array = (as2_msgs__msg__ControlMode__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__ControlMode__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__ControlMode__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__ControlMode__Sequence__destroy(as2_msgs__msg__ControlMode__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__ControlMode__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__ControlMode__Sequence__are_equal(const as2_msgs__msg__ControlMode__Sequence * lhs, const as2_msgs__msg__ControlMode__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__ControlMode__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__ControlMode__Sequence__copy(
  const as2_msgs__msg__ControlMode__Sequence * input,
  as2_msgs__msg__ControlMode__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__ControlMode);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__ControlMode * data =
      (as2_msgs__msg__ControlMode *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__ControlMode__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__ControlMode__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__ControlMode__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
