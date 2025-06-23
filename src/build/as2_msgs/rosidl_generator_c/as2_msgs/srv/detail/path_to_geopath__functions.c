// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:srv/PathToGeopath.idl
// generated code does not contain a copyright notice
#include "as2_msgs/srv/detail/path_to_geopath__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `path`
#include "nav_msgs/msg/detail/path__functions.h"

bool
as2_msgs__srv__PathToGeopath_Request__init(as2_msgs__srv__PathToGeopath_Request * msg)
{
  if (!msg) {
    return false;
  }
  // path
  if (!nav_msgs__msg__Path__init(&msg->path)) {
    as2_msgs__srv__PathToGeopath_Request__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__srv__PathToGeopath_Request__fini(as2_msgs__srv__PathToGeopath_Request * msg)
{
  if (!msg) {
    return;
  }
  // path
  nav_msgs__msg__Path__fini(&msg->path);
}

bool
as2_msgs__srv__PathToGeopath_Request__are_equal(const as2_msgs__srv__PathToGeopath_Request * lhs, const as2_msgs__srv__PathToGeopath_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // path
  if (!nav_msgs__msg__Path__are_equal(
      &(lhs->path), &(rhs->path)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__srv__PathToGeopath_Request__copy(
  const as2_msgs__srv__PathToGeopath_Request * input,
  as2_msgs__srv__PathToGeopath_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // path
  if (!nav_msgs__msg__Path__copy(
      &(input->path), &(output->path)))
  {
    return false;
  }
  return true;
}

as2_msgs__srv__PathToGeopath_Request *
as2_msgs__srv__PathToGeopath_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__PathToGeopath_Request * msg = (as2_msgs__srv__PathToGeopath_Request *)allocator.allocate(sizeof(as2_msgs__srv__PathToGeopath_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__srv__PathToGeopath_Request));
  bool success = as2_msgs__srv__PathToGeopath_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__srv__PathToGeopath_Request__destroy(as2_msgs__srv__PathToGeopath_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__srv__PathToGeopath_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__srv__PathToGeopath_Request__Sequence__init(as2_msgs__srv__PathToGeopath_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__PathToGeopath_Request * data = NULL;

  if (size) {
    data = (as2_msgs__srv__PathToGeopath_Request *)allocator.zero_allocate(size, sizeof(as2_msgs__srv__PathToGeopath_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__srv__PathToGeopath_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__srv__PathToGeopath_Request__fini(&data[i - 1]);
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
as2_msgs__srv__PathToGeopath_Request__Sequence__fini(as2_msgs__srv__PathToGeopath_Request__Sequence * array)
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
      as2_msgs__srv__PathToGeopath_Request__fini(&array->data[i]);
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

as2_msgs__srv__PathToGeopath_Request__Sequence *
as2_msgs__srv__PathToGeopath_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__PathToGeopath_Request__Sequence * array = (as2_msgs__srv__PathToGeopath_Request__Sequence *)allocator.allocate(sizeof(as2_msgs__srv__PathToGeopath_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__srv__PathToGeopath_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__srv__PathToGeopath_Request__Sequence__destroy(as2_msgs__srv__PathToGeopath_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__srv__PathToGeopath_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__srv__PathToGeopath_Request__Sequence__are_equal(const as2_msgs__srv__PathToGeopath_Request__Sequence * lhs, const as2_msgs__srv__PathToGeopath_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__srv__PathToGeopath_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__srv__PathToGeopath_Request__Sequence__copy(
  const as2_msgs__srv__PathToGeopath_Request__Sequence * input,
  as2_msgs__srv__PathToGeopath_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__srv__PathToGeopath_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__srv__PathToGeopath_Request * data =
      (as2_msgs__srv__PathToGeopath_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__srv__PathToGeopath_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__srv__PathToGeopath_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__srv__PathToGeopath_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `geo_path`
#include "geographic_msgs/msg/detail/geo_path__functions.h"

bool
as2_msgs__srv__PathToGeopath_Response__init(as2_msgs__srv__PathToGeopath_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // geo_path
  if (!geographic_msgs__msg__GeoPath__init(&msg->geo_path)) {
    as2_msgs__srv__PathToGeopath_Response__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__srv__PathToGeopath_Response__fini(as2_msgs__srv__PathToGeopath_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // geo_path
  geographic_msgs__msg__GeoPath__fini(&msg->geo_path);
}

bool
as2_msgs__srv__PathToGeopath_Response__are_equal(const as2_msgs__srv__PathToGeopath_Response * lhs, const as2_msgs__srv__PathToGeopath_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // geo_path
  if (!geographic_msgs__msg__GeoPath__are_equal(
      &(lhs->geo_path), &(rhs->geo_path)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__srv__PathToGeopath_Response__copy(
  const as2_msgs__srv__PathToGeopath_Response * input,
  as2_msgs__srv__PathToGeopath_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // geo_path
  if (!geographic_msgs__msg__GeoPath__copy(
      &(input->geo_path), &(output->geo_path)))
  {
    return false;
  }
  return true;
}

as2_msgs__srv__PathToGeopath_Response *
as2_msgs__srv__PathToGeopath_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__PathToGeopath_Response * msg = (as2_msgs__srv__PathToGeopath_Response *)allocator.allocate(sizeof(as2_msgs__srv__PathToGeopath_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__srv__PathToGeopath_Response));
  bool success = as2_msgs__srv__PathToGeopath_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__srv__PathToGeopath_Response__destroy(as2_msgs__srv__PathToGeopath_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__srv__PathToGeopath_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__srv__PathToGeopath_Response__Sequence__init(as2_msgs__srv__PathToGeopath_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__PathToGeopath_Response * data = NULL;

  if (size) {
    data = (as2_msgs__srv__PathToGeopath_Response *)allocator.zero_allocate(size, sizeof(as2_msgs__srv__PathToGeopath_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__srv__PathToGeopath_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__srv__PathToGeopath_Response__fini(&data[i - 1]);
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
as2_msgs__srv__PathToGeopath_Response__Sequence__fini(as2_msgs__srv__PathToGeopath_Response__Sequence * array)
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
      as2_msgs__srv__PathToGeopath_Response__fini(&array->data[i]);
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

as2_msgs__srv__PathToGeopath_Response__Sequence *
as2_msgs__srv__PathToGeopath_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__PathToGeopath_Response__Sequence * array = (as2_msgs__srv__PathToGeopath_Response__Sequence *)allocator.allocate(sizeof(as2_msgs__srv__PathToGeopath_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__srv__PathToGeopath_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__srv__PathToGeopath_Response__Sequence__destroy(as2_msgs__srv__PathToGeopath_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__srv__PathToGeopath_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__srv__PathToGeopath_Response__Sequence__are_equal(const as2_msgs__srv__PathToGeopath_Response__Sequence * lhs, const as2_msgs__srv__PathToGeopath_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__srv__PathToGeopath_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__srv__PathToGeopath_Response__Sequence__copy(
  const as2_msgs__srv__PathToGeopath_Response__Sequence * input,
  as2_msgs__srv__PathToGeopath_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__srv__PathToGeopath_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__srv__PathToGeopath_Response * data =
      (as2_msgs__srv__PathToGeopath_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__srv__PathToGeopath_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__srv__PathToGeopath_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__srv__PathToGeopath_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
