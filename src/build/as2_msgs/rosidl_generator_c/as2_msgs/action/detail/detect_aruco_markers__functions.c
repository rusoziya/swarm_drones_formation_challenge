// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from as2_msgs:action/DetectArucoMarkers.idl
// generated code does not contain a copyright notice
#include "as2_msgs/action/detail/detect_aruco_markers__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `target_ids`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
as2_msgs__action__DetectArucoMarkers_Goal__init(as2_msgs__action__DetectArucoMarkers_Goal * msg)
{
  if (!msg) {
    return false;
  }
  // target_ids
  if (!rosidl_runtime_c__uint16__Sequence__init(&msg->target_ids, 0)) {
    as2_msgs__action__DetectArucoMarkers_Goal__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_Goal__fini(as2_msgs__action__DetectArucoMarkers_Goal * msg)
{
  if (!msg) {
    return;
  }
  // target_ids
  rosidl_runtime_c__uint16__Sequence__fini(&msg->target_ids);
}

bool
as2_msgs__action__DetectArucoMarkers_Goal__are_equal(const as2_msgs__action__DetectArucoMarkers_Goal * lhs, const as2_msgs__action__DetectArucoMarkers_Goal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // target_ids
  if (!rosidl_runtime_c__uint16__Sequence__are_equal(
      &(lhs->target_ids), &(rhs->target_ids)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_Goal__copy(
  const as2_msgs__action__DetectArucoMarkers_Goal * input,
  as2_msgs__action__DetectArucoMarkers_Goal * output)
{
  if (!input || !output) {
    return false;
  }
  // target_ids
  if (!rosidl_runtime_c__uint16__Sequence__copy(
      &(input->target_ids), &(output->target_ids)))
  {
    return false;
  }
  return true;
}

as2_msgs__action__DetectArucoMarkers_Goal *
as2_msgs__action__DetectArucoMarkers_Goal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Goal * msg = (as2_msgs__action__DetectArucoMarkers_Goal *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_Goal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_Goal));
  bool success = as2_msgs__action__DetectArucoMarkers_Goal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_Goal__destroy(as2_msgs__action__DetectArucoMarkers_Goal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_Goal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_Goal__Sequence__init(as2_msgs__action__DetectArucoMarkers_Goal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Goal * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_Goal *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_Goal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_Goal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_Goal__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_Goal__Sequence__fini(as2_msgs__action__DetectArucoMarkers_Goal__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_Goal__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_Goal__Sequence *
as2_msgs__action__DetectArucoMarkers_Goal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Goal__Sequence * array = (as2_msgs__action__DetectArucoMarkers_Goal__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_Goal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_Goal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_Goal__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_Goal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_Goal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_Goal__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_Goal__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_Goal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_Goal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_Goal__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_Goal__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_Goal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_Goal);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_Goal * data =
      (as2_msgs__action__DetectArucoMarkers_Goal *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_Goal__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_Goal__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_Goal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
as2_msgs__action__DetectArucoMarkers_Result__init(as2_msgs__action__DetectArucoMarkers_Result * msg)
{
  if (!msg) {
    return false;
  }
  // success
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_Result__fini(as2_msgs__action__DetectArucoMarkers_Result * msg)
{
  if (!msg) {
    return;
  }
  // success
}

bool
as2_msgs__action__DetectArucoMarkers_Result__are_equal(const as2_msgs__action__DetectArucoMarkers_Result * lhs, const as2_msgs__action__DetectArucoMarkers_Result * rhs)
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
as2_msgs__action__DetectArucoMarkers_Result__copy(
  const as2_msgs__action__DetectArucoMarkers_Result * input,
  as2_msgs__action__DetectArucoMarkers_Result * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  return true;
}

as2_msgs__action__DetectArucoMarkers_Result *
as2_msgs__action__DetectArucoMarkers_Result__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Result * msg = (as2_msgs__action__DetectArucoMarkers_Result *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_Result), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_Result));
  bool success = as2_msgs__action__DetectArucoMarkers_Result__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_Result__destroy(as2_msgs__action__DetectArucoMarkers_Result * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_Result__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_Result__Sequence__init(as2_msgs__action__DetectArucoMarkers_Result__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Result * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_Result *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_Result), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_Result__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_Result__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_Result__Sequence__fini(as2_msgs__action__DetectArucoMarkers_Result__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_Result__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_Result__Sequence *
as2_msgs__action__DetectArucoMarkers_Result__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Result__Sequence * array = (as2_msgs__action__DetectArucoMarkers_Result__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_Result__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_Result__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_Result__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_Result__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_Result__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_Result__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_Result__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_Result__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_Result__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_Result__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_Result__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_Result__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_Result);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_Result * data =
      (as2_msgs__action__DetectArucoMarkers_Result *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_Result__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_Result__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_Result__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
as2_msgs__action__DetectArucoMarkers_Feedback__init(as2_msgs__action__DetectArucoMarkers_Feedback * msg)
{
  if (!msg) {
    return false;
  }
  // sucess
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_Feedback__fini(as2_msgs__action__DetectArucoMarkers_Feedback * msg)
{
  if (!msg) {
    return;
  }
  // sucess
}

bool
as2_msgs__action__DetectArucoMarkers_Feedback__are_equal(const as2_msgs__action__DetectArucoMarkers_Feedback * lhs, const as2_msgs__action__DetectArucoMarkers_Feedback * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // sucess
  if (lhs->sucess != rhs->sucess) {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_Feedback__copy(
  const as2_msgs__action__DetectArucoMarkers_Feedback * input,
  as2_msgs__action__DetectArucoMarkers_Feedback * output)
{
  if (!input || !output) {
    return false;
  }
  // sucess
  output->sucess = input->sucess;
  return true;
}

as2_msgs__action__DetectArucoMarkers_Feedback *
as2_msgs__action__DetectArucoMarkers_Feedback__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Feedback * msg = (as2_msgs__action__DetectArucoMarkers_Feedback *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_Feedback), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_Feedback));
  bool success = as2_msgs__action__DetectArucoMarkers_Feedback__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_Feedback__destroy(as2_msgs__action__DetectArucoMarkers_Feedback * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_Feedback__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__init(as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Feedback * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_Feedback *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_Feedback), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_Feedback__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_Feedback__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__fini(as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_Feedback__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_Feedback__Sequence *
as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * array = (as2_msgs__action__DetectArucoMarkers_Feedback__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_Feedback__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_Feedback__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_Feedback__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_Feedback__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_Feedback);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_Feedback * data =
      (as2_msgs__action__DetectArucoMarkers_Feedback *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_Feedback__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_Feedback__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_Feedback__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
#include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `goal`
// already included above
// #include "as2_msgs/action/detail/detect_aruco_markers__functions.h"

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__init(as2_msgs__action__DetectArucoMarkers_SendGoal_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(msg);
    return false;
  }
  // goal
  if (!as2_msgs__action__DetectArucoMarkers_Goal__init(&msg->goal)) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(as2_msgs__action__DetectArucoMarkers_SendGoal_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // goal
  as2_msgs__action__DetectArucoMarkers_Goal__fini(&msg->goal);
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__are_equal(const as2_msgs__action__DetectArucoMarkers_SendGoal_Request * lhs, const as2_msgs__action__DetectArucoMarkers_SendGoal_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // goal
  if (!as2_msgs__action__DetectArucoMarkers_Goal__are_equal(
      &(lhs->goal), &(rhs->goal)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__copy(
  const as2_msgs__action__DetectArucoMarkers_SendGoal_Request * input,
  as2_msgs__action__DetectArucoMarkers_SendGoal_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // goal
  if (!as2_msgs__action__DetectArucoMarkers_Goal__copy(
      &(input->goal), &(output->goal)))
  {
    return false;
  }
  return true;
}

as2_msgs__action__DetectArucoMarkers_SendGoal_Request *
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_SendGoal_Request * msg = (as2_msgs__action__DetectArucoMarkers_SendGoal_Request *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Request));
  bool success = as2_msgs__action__DetectArucoMarkers_SendGoal_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__destroy(as2_msgs__action__DetectArucoMarkers_SendGoal_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__init(as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_SendGoal_Request * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_SendGoal_Request *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_SendGoal_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__fini(as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence *
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * array = (as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_SendGoal_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_SendGoal_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_SendGoal_Request * data =
      (as2_msgs__action__DetectArucoMarkers_SendGoal_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_SendGoal_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_SendGoal_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_SendGoal_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__init(as2_msgs__action__DetectArucoMarkers_SendGoal_Response * msg)
{
  if (!msg) {
    return false;
  }
  // accepted
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Response__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__fini(as2_msgs__action__DetectArucoMarkers_SendGoal_Response * msg)
{
  if (!msg) {
    return;
  }
  // accepted
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__are_equal(const as2_msgs__action__DetectArucoMarkers_SendGoal_Response * lhs, const as2_msgs__action__DetectArucoMarkers_SendGoal_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // accepted
  if (lhs->accepted != rhs->accepted) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__copy(
  const as2_msgs__action__DetectArucoMarkers_SendGoal_Response * input,
  as2_msgs__action__DetectArucoMarkers_SendGoal_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // accepted
  output->accepted = input->accepted;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

as2_msgs__action__DetectArucoMarkers_SendGoal_Response *
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_SendGoal_Response * msg = (as2_msgs__action__DetectArucoMarkers_SendGoal_Response *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Response));
  bool success = as2_msgs__action__DetectArucoMarkers_SendGoal_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__destroy(as2_msgs__action__DetectArucoMarkers_SendGoal_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__init(as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_SendGoal_Response * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_SendGoal_Response *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_SendGoal_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_SendGoal_Response__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__fini(as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_SendGoal_Response__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence *
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * array = (as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_SendGoal_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_SendGoal_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_SendGoal_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_SendGoal_Response * data =
      (as2_msgs__action__DetectArucoMarkers_SendGoal_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_SendGoal_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_SendGoal_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_SendGoal_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Request__init(as2_msgs__action__DetectArucoMarkers_GetResult_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    as2_msgs__action__DetectArucoMarkers_GetResult_Request__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_GetResult_Request__fini(as2_msgs__action__DetectArucoMarkers_GetResult_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Request__are_equal(const as2_msgs__action__DetectArucoMarkers_GetResult_Request * lhs, const as2_msgs__action__DetectArucoMarkers_GetResult_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Request__copy(
  const as2_msgs__action__DetectArucoMarkers_GetResult_Request * input,
  as2_msgs__action__DetectArucoMarkers_GetResult_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  return true;
}

as2_msgs__action__DetectArucoMarkers_GetResult_Request *
as2_msgs__action__DetectArucoMarkers_GetResult_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_GetResult_Request * msg = (as2_msgs__action__DetectArucoMarkers_GetResult_Request *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Request));
  bool success = as2_msgs__action__DetectArucoMarkers_GetResult_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_GetResult_Request__destroy(as2_msgs__action__DetectArucoMarkers_GetResult_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_GetResult_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__init(as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_GetResult_Request * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_GetResult_Request *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_GetResult_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_GetResult_Request__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__fini(as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_GetResult_Request__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence *
as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * array = (as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_GetResult_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_GetResult_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_GetResult_Request * data =
      (as2_msgs__action__DetectArucoMarkers_GetResult_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_GetResult_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_GetResult_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_GetResult_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `result`
// already included above
// #include "as2_msgs/action/detail/detect_aruco_markers__functions.h"

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Response__init(as2_msgs__action__DetectArucoMarkers_GetResult_Response * msg)
{
  if (!msg) {
    return false;
  }
  // status
  // result
  if (!as2_msgs__action__DetectArucoMarkers_Result__init(&msg->result)) {
    as2_msgs__action__DetectArucoMarkers_GetResult_Response__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_GetResult_Response__fini(as2_msgs__action__DetectArucoMarkers_GetResult_Response * msg)
{
  if (!msg) {
    return;
  }
  // status
  // result
  as2_msgs__action__DetectArucoMarkers_Result__fini(&msg->result);
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Response__are_equal(const as2_msgs__action__DetectArucoMarkers_GetResult_Response * lhs, const as2_msgs__action__DetectArucoMarkers_GetResult_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  // result
  if (!as2_msgs__action__DetectArucoMarkers_Result__are_equal(
      &(lhs->result), &(rhs->result)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Response__copy(
  const as2_msgs__action__DetectArucoMarkers_GetResult_Response * input,
  as2_msgs__action__DetectArucoMarkers_GetResult_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // status
  output->status = input->status;
  // result
  if (!as2_msgs__action__DetectArucoMarkers_Result__copy(
      &(input->result), &(output->result)))
  {
    return false;
  }
  return true;
}

as2_msgs__action__DetectArucoMarkers_GetResult_Response *
as2_msgs__action__DetectArucoMarkers_GetResult_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_GetResult_Response * msg = (as2_msgs__action__DetectArucoMarkers_GetResult_Response *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Response));
  bool success = as2_msgs__action__DetectArucoMarkers_GetResult_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_GetResult_Response__destroy(as2_msgs__action__DetectArucoMarkers_GetResult_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_GetResult_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__init(as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_GetResult_Response * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_GetResult_Response *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_GetResult_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_GetResult_Response__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__fini(as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_GetResult_Response__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence *
as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * array = (as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_GetResult_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_GetResult_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_GetResult_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_GetResult_Response * data =
      (as2_msgs__action__DetectArucoMarkers_GetResult_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_GetResult_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_GetResult_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_GetResult_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `feedback`
// already included above
// #include "as2_msgs/action/detail/detect_aruco_markers__functions.h"

bool
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__init(as2_msgs__action__DetectArucoMarkers_FeedbackMessage * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(msg);
    return false;
  }
  // feedback
  if (!as2_msgs__action__DetectArucoMarkers_Feedback__init(&msg->feedback)) {
    as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(msg);
    return false;
  }
  return true;
}

void
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(as2_msgs__action__DetectArucoMarkers_FeedbackMessage * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // feedback
  as2_msgs__action__DetectArucoMarkers_Feedback__fini(&msg->feedback);
}

bool
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__are_equal(const as2_msgs__action__DetectArucoMarkers_FeedbackMessage * lhs, const as2_msgs__action__DetectArucoMarkers_FeedbackMessage * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // feedback
  if (!as2_msgs__action__DetectArucoMarkers_Feedback__are_equal(
      &(lhs->feedback), &(rhs->feedback)))
  {
    return false;
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__copy(
  const as2_msgs__action__DetectArucoMarkers_FeedbackMessage * input,
  as2_msgs__action__DetectArucoMarkers_FeedbackMessage * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // feedback
  if (!as2_msgs__action__DetectArucoMarkers_Feedback__copy(
      &(input->feedback), &(output->feedback)))
  {
    return false;
  }
  return true;
}

as2_msgs__action__DetectArucoMarkers_FeedbackMessage *
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_FeedbackMessage * msg = (as2_msgs__action__DetectArucoMarkers_FeedbackMessage *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_FeedbackMessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(as2_msgs__action__DetectArucoMarkers_FeedbackMessage));
  bool success = as2_msgs__action__DetectArucoMarkers_FeedbackMessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__destroy(as2_msgs__action__DetectArucoMarkers_FeedbackMessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__init(as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_FeedbackMessage * data = NULL;

  if (size) {
    data = (as2_msgs__action__DetectArucoMarkers_FeedbackMessage *)allocator.zero_allocate(size, sizeof(as2_msgs__action__DetectArucoMarkers_FeedbackMessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = as2_msgs__action__DetectArucoMarkers_FeedbackMessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(&data[i - 1]);
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
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__fini(as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * array)
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
      as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(&array->data[i]);
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

as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence *
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * array = (as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence *)allocator.allocate(sizeof(as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__destroy(as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__are_equal(const as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * lhs, const as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_FeedbackMessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence__copy(
  const as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * input,
  as2_msgs__action__DetectArucoMarkers_FeedbackMessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(as2_msgs__action__DetectArucoMarkers_FeedbackMessage);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    as2_msgs__action__DetectArucoMarkers_FeedbackMessage * data =
      (as2_msgs__action__DetectArucoMarkers_FeedbackMessage *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!as2_msgs__action__DetectArucoMarkers_FeedbackMessage__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          as2_msgs__action__DetectArucoMarkers_FeedbackMessage__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!as2_msgs__action__DetectArucoMarkers_FeedbackMessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
