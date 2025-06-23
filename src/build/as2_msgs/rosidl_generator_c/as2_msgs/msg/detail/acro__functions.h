// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from as2_msgs:msg/Acro.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__MSG__DETAIL__ACRO__FUNCTIONS_H_
#define AS2_MSGS__MSG__DETAIL__ACRO__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "as2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "as2_msgs/msg/detail/acro__struct.h"

/// Initialize msg/Acro message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * as2_msgs__msg__Acro
 * )) before or use
 * as2_msgs__msg__Acro__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__msg__Acro__init(as2_msgs__msg__Acro * msg);

/// Finalize msg/Acro message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__msg__Acro__fini(as2_msgs__msg__Acro * msg);

/// Create msg/Acro message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * as2_msgs__msg__Acro__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
as2_msgs__msg__Acro *
as2_msgs__msg__Acro__create();

/// Destroy msg/Acro message.
/**
 * It calls
 * as2_msgs__msg__Acro__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__msg__Acro__destroy(as2_msgs__msg__Acro * msg);

/// Check for msg/Acro message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__msg__Acro__are_equal(const as2_msgs__msg__Acro * lhs, const as2_msgs__msg__Acro * rhs);

/// Copy a msg/Acro message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__msg__Acro__copy(
  const as2_msgs__msg__Acro * input,
  as2_msgs__msg__Acro * output);

/// Initialize array of msg/Acro messages.
/**
 * It allocates the memory for the number of elements and calls
 * as2_msgs__msg__Acro__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__msg__Acro__Sequence__init(as2_msgs__msg__Acro__Sequence * array, size_t size);

/// Finalize array of msg/Acro messages.
/**
 * It calls
 * as2_msgs__msg__Acro__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__msg__Acro__Sequence__fini(as2_msgs__msg__Acro__Sequence * array);

/// Create array of msg/Acro messages.
/**
 * It allocates the memory for the array and calls
 * as2_msgs__msg__Acro__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
as2_msgs__msg__Acro__Sequence *
as2_msgs__msg__Acro__Sequence__create(size_t size);

/// Destroy array of msg/Acro messages.
/**
 * It calls
 * as2_msgs__msg__Acro__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__msg__Acro__Sequence__destroy(as2_msgs__msg__Acro__Sequence * array);

/// Check for msg/Acro message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__msg__Acro__Sequence__are_equal(const as2_msgs__msg__Acro__Sequence * lhs, const as2_msgs__msg__Acro__Sequence * rhs);

/// Copy an array of msg/Acro messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__msg__Acro__Sequence__copy(
  const as2_msgs__msg__Acro__Sequence * input,
  as2_msgs__msg__Acro__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__MSG__DETAIL__ACRO__FUNCTIONS_H_
