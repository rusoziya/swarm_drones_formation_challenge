// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:msg/Geozone.idl
// generated code does not contain a copyright notice
#include "as2_msgs/msg/detail/geozone__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `type`
// Member `data_type`
#include "rosidl_runtime_c/string_functions.h"
// Member `polygon`
#include "geometry_msgs/msg/detail/polygon__functions.h"

bool
as2_msgs__msg__Geozone__init(as2_msgs__msg__Geozone * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // alert
  // type
  if (!rosidl_runtime_c__String__init(&msg->type)) {
    as2_msgs__msg__Geozone__fini(msg);
    return false;
  }
  // data_type
  if (!rosidl_runtime_c__String__init(&msg->data_type)) {
    as2_msgs__msg__Geozone__fini(msg);
    return false;
  }
  // polygon
  if (!geometry_msgs__msg__Polygon__init(&msg->polygon)) {
    as2_msgs__msg__Geozone__fini(msg);
    return false;
  }
  // z_up
  // z_down
  return true;
}

void
as2_msgs__msg__Geozone__fini(as2_msgs__msg__Geozone * msg)
{
  if (!msg) {
    return;
  }
  // id
  // alert
  // type
  rosidl_runtime_c__String__fini(&msg->type);
  // data_type
  rosidl_runtime_c__String__fini(&msg->data_type);
  // polygon
  geometry_msgs__msg__Polygon__fini(&msg->polygon);
  // z_up
  // z_down
}

bool
as2_msgs__msg__Geozone__are_equal(const as2_msgs__msg__Geozone * lhs, const as2_msgs__msg__Geozone * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // alert
  if (lhs->alert != rhs->alert) {
    return false;
  }
  // type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->type), &(rhs->type)))
  {
    return false;
  }
  // data_type
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->data_type), &(rhs->data_type)))
  {
    return false;
  }
  // polygon
  if (!geometry_msgs__msg__Polygon__are_equal(
      &(lhs->polygon), &(rhs->polygon)))
  {
    return false;
  }
  // z_up
  if (lhs->z_up != rhs->z_up) {
    return false;
  }
  // z_down
  if (lhs->z_down != rhs->z_down) {
    return false;
  }
  return true;
}

bool
as2_msgs__msg__Geozone__copy(
  const as2_msgs__msg__Geozone * input,
  as2_msgs__msg__Geozone * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // alert
  output->alert = input->alert;
  // type
  if (!rosidl_runtime_c__String__copy(
      &(input->type), &(output->type)))
  {
    return false;
  }
  // data_type
  if (!rosidl_runtime_c__String__copy(
      &(input->data_type), &(output->data_type)))
  {
    return false;
  }
  // polygon
  if (!geometry_msgs__msg__Polygon__copy(
      &(input->polygon), &(output->polygon)))
  {
    return false;
  }
  // z_up
  output->z_up = input->z_up;
  // z_down
  output->z_down = input->z_down;
  return true;
}

as2_msgs__msg__Geozone *
as2_msgs__msg__Geozone__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__Geozone * msg = (as2_msgs__msg__Geozone *)allocator.allocate(sizeof(as2_msgs__msg__Geozone), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__msg__Geozone));
  bool success = as2_msgs__msg__Geozone__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__msg__Geozone__destroy(as2_msgs__msg__Geozone * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__msg__Geozone__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__msg__Geozone__Sequence__init(as2_msgs__msg__Geozone__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__Geozone * data = NULL;

  if (size) {
    data = (as2_msgs__msg__Geozone *)allocator.zero_allocate(size, sizeof(as2_msgs__msg__Geozone), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__msg__Geozone__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__msg__Geozone__fini(&data[i - 1]);
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
as2_msgs__msg__Geozone__Sequence__fini(as2_msgs__msg__Geozone__Sequence * array)
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
      as2_msgs__msg__Geozone__fini(&array->data[i]);
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

as2_msgs__msg__Geozone__Sequence *
as2_msgs__msg__Geozone__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__msg__Geozone__Sequence * array = (as2_msgs__msg__Geozone__Sequence *)allocator.allocate(sizeof(as2_msgs__msg__Geozone__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__msg__Geozone__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__msg__Geozone__Sequence__destroy(as2_msgs__msg__Geozone__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__msg__Geozone__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__msg__Geozone__Sequence__are_equal(const as2_msgs__msg__Geozone__Sequence * lhs, const as2_msgs__msg__Geozone__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__msg__Geozone__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__msg__Geozone__Sequence__copy(
  const as2_msgs__msg__Geozone__Sequence * input,
  as2_msgs__msg__Geozone__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__msg__Geozone);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__msg__Geozone * data =
      (as2_msgs__msg__Geozone *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__msg__Geozone__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__msg__Geozone__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__msg__Geozone__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
