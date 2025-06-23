// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:srv/AddStaticTransformGps.idl
// generated code does not contain a copyright notice
#include "as2_msgs/srv/detail/add_static_transform_gps__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `frame_id`
// Member `child_frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `gps_position`
#include "sensor_msgs/msg/detail/nav_sat_fix__functions.h"

bool
as2_msgs__srv__AddStaticTransformGps_Request__init(as2_msgs__srv__AddStaticTransformGps_Request * msg)
{
  if (!msg) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__init(&msg->frame_id)) {
    as2_msgs__srv__AddStaticTransformGps_Request__fini(msg);
    return false;
  }
  // child_frame_id
  if (!rosidl_runtime_c__String__init(&msg->child_frame_id)) {
    as2_msgs__srv__AddStaticTransformGps_Request__fini(msg);
    return false;
  }
  // gps_position
  if (!sensor_msgs__msg__NavSatFix__init(&msg->gps_position)) {
    as2_msgs__srv__AddStaticTransformGps_Request__fini(msg);
    return false;
  }
  // azimuth
  // elevation
  // bank
  return true;
}

void
as2_msgs__srv__AddStaticTransformGps_Request__fini(as2_msgs__srv__AddStaticTransformGps_Request * msg)
{
  if (!msg) {
    return;
  }
  // frame_id
  rosidl_runtime_c__String__fini(&msg->frame_id);
  // child_frame_id
  rosidl_runtime_c__String__fini(&msg->child_frame_id);
  // gps_position
  sensor_msgs__msg__NavSatFix__fini(&msg->gps_position);
  // azimuth
  // elevation
  // bank
}

bool
as2_msgs__srv__AddStaticTransformGps_Request__are_equal(const as2_msgs__srv__AddStaticTransformGps_Request * lhs, const as2_msgs__srv__AddStaticTransformGps_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->frame_id), &(rhs->frame_id)))
  {
    return false;
  }
  // child_frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->child_frame_id), &(rhs->child_frame_id)))
  {
    return false;
  }
  // gps_position
  if (!sensor_msgs__msg__NavSatFix__are_equal(
      &(lhs->gps_position), &(rhs->gps_position)))
  {
    return false;
  }
  // azimuth
  if (lhs->azimuth != rhs->azimuth) {
    return false;
  }
  // elevation
  if (lhs->elevation != rhs->elevation) {
    return false;
  }
  // bank
  if (lhs->bank != rhs->bank) {
    return false;
  }
  return true;
}

bool
as2_msgs__srv__AddStaticTransformGps_Request__copy(
  const as2_msgs__srv__AddStaticTransformGps_Request * input,
  as2_msgs__srv__AddStaticTransformGps_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->frame_id), &(output->frame_id)))
  {
    return false;
  }
  // child_frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->child_frame_id), &(output->child_frame_id)))
  {
    return false;
  }
  // gps_position
  if (!sensor_msgs__msg__NavSatFix__copy(
      &(input->gps_position), &(output->gps_position)))
  {
    return false;
  }
  // azimuth
  output->azimuth = input->azimuth;
  // elevation
  output->elevation = input->elevation;
  // bank
  output->bank = input->bank;
  return true;
}

as2_msgs__srv__AddStaticTransformGps_Request *
as2_msgs__srv__AddStaticTransformGps_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__AddStaticTransformGps_Request * msg = (as2_msgs__srv__AddStaticTransformGps_Request *)allocator.allocate(sizeof(as2_msgs__srv__AddStaticTransformGps_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__srv__AddStaticTransformGps_Request));
  bool success = as2_msgs__srv__AddStaticTransformGps_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__srv__AddStaticTransformGps_Request__destroy(as2_msgs__srv__AddStaticTransformGps_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__srv__AddStaticTransformGps_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__srv__AddStaticTransformGps_Request__Sequence__init(as2_msgs__srv__AddStaticTransformGps_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__AddStaticTransformGps_Request * data = NULL;

  if (size) {
    data = (as2_msgs__srv__AddStaticTransformGps_Request *)allocator.zero_allocate(size, sizeof(as2_msgs__srv__AddStaticTransformGps_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__srv__AddStaticTransformGps_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__srv__AddStaticTransformGps_Request__fini(&data[i - 1]);
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
as2_msgs__srv__AddStaticTransformGps_Request__Sequence__fini(as2_msgs__srv__AddStaticTransformGps_Request__Sequence * array)
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
      as2_msgs__srv__AddStaticTransformGps_Request__fini(&array->data[i]);
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

as2_msgs__srv__AddStaticTransformGps_Request__Sequence *
as2_msgs__srv__AddStaticTransformGps_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__AddStaticTransformGps_Request__Sequence * array = (as2_msgs__srv__AddStaticTransformGps_Request__Sequence *)allocator.allocate(sizeof(as2_msgs__srv__AddStaticTransformGps_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__srv__AddStaticTransformGps_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__srv__AddStaticTransformGps_Request__Sequence__destroy(as2_msgs__srv__AddStaticTransformGps_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__srv__AddStaticTransformGps_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__srv__AddStaticTransformGps_Request__Sequence__are_equal(const as2_msgs__srv__AddStaticTransformGps_Request__Sequence * lhs, const as2_msgs__srv__AddStaticTransformGps_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__srv__AddStaticTransformGps_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__srv__AddStaticTransformGps_Request__Sequence__copy(
  const as2_msgs__srv__AddStaticTransformGps_Request__Sequence * input,
  as2_msgs__srv__AddStaticTransformGps_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__srv__AddStaticTransformGps_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__srv__AddStaticTransformGps_Request * data =
      (as2_msgs__srv__AddStaticTransformGps_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__srv__AddStaticTransformGps_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__srv__AddStaticTransformGps_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__srv__AddStaticTransformGps_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
as2_msgs__srv__AddStaticTransformGps_Response__init(as2_msgs__srv__AddStaticTransformGps_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  return true;
}

void
as2_msgs__srv__AddStaticTransformGps_Response__fini(as2_msgs__srv__AddStaticTransformGps_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
}

bool
as2_msgs__srv__AddStaticTransformGps_Response__are_equal(const as2_msgs__srv__AddStaticTransformGps_Response * lhs, const as2_msgs__srv__AddStaticTransformGps_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  return true;
}

bool
as2_msgs__srv__AddStaticTransformGps_Response__copy(
  const as2_msgs__srv__AddStaticTransformGps_Response * input,
  as2_msgs__srv__AddStaticTransformGps_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  return true;
}

as2_msgs__srv__AddStaticTransformGps_Response *
as2_msgs__srv__AddStaticTransformGps_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__AddStaticTransformGps_Response * msg = (as2_msgs__srv__AddStaticTransformGps_Response *)allocator.allocate(sizeof(as2_msgs__srv__AddStaticTransformGps_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__srv__AddStaticTransformGps_Response));
  bool success = as2_msgs__srv__AddStaticTransformGps_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__srv__AddStaticTransformGps_Response__destroy(as2_msgs__srv__AddStaticTransformGps_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__srv__AddStaticTransformGps_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__srv__AddStaticTransformGps_Response__Sequence__init(as2_msgs__srv__AddStaticTransformGps_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__AddStaticTransformGps_Response * data = NULL;

  if (size) {
    data = (as2_msgs__srv__AddStaticTransformGps_Response *)allocator.zero_allocate(size, sizeof(as2_msgs__srv__AddStaticTransformGps_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__srv__AddStaticTransformGps_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__srv__AddStaticTransformGps_Response__fini(&data[i - 1]);
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
as2_msgs__srv__AddStaticTransformGps_Response__Sequence__fini(as2_msgs__srv__AddStaticTransformGps_Response__Sequence * array)
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
      as2_msgs__srv__AddStaticTransformGps_Response__fini(&array->data[i]);
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

as2_msgs__srv__AddStaticTransformGps_Response__Sequence *
as2_msgs__srv__AddStaticTransformGps_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__srv__AddStaticTransformGps_Response__Sequence * array = (as2_msgs__srv__AddStaticTransformGps_Response__Sequence *)allocator.allocate(sizeof(as2_msgs__srv__AddStaticTransformGps_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__srv__AddStaticTransformGps_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__srv__AddStaticTransformGps_Response__Sequence__destroy(as2_msgs__srv__AddStaticTransformGps_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__srv__AddStaticTransformGps_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__srv__AddStaticTransformGps_Response__Sequence__are_equal(const as2_msgs__srv__AddStaticTransformGps_Response__Sequence * lhs, const as2_msgs__srv__AddStaticTransformGps_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__srv__AddStaticTransformGps_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__srv__AddStaticTransformGps_Response__Sequence__copy(
  const as2_msgs__srv__AddStaticTransformGps_Response__Sequence * input,
  as2_msgs__srv__AddStaticTransformGps_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__srv__AddStaticTransformGps_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__srv__AddStaticTransformGps_Response * data =
      (as2_msgs__srv__AddStaticTransformGps_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__srv__AddStaticTransformGps_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__srv__AddStaticTransformGps_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__srv__AddStaticTransformGps_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
