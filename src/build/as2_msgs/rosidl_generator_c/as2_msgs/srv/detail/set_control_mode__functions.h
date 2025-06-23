// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from as2_msgs:srv/SetControlMode.idl
// generated code does not contain a copyright notice

#ifndef AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__FUNCTIONS_H_
#define AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "as2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "as2_msgs/srv/detail/set_control_mode__struct.h"

/// Initialize srv/SetControlMode message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * as2_msgs__srv__SetControlMode_Request
 * )) before or use
 * as2_msgs__srv__SetControlMode_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Request__init(as2_msgs__srv__SetControlMode_Request * msg);

/// Finalize srv/SetControlMode message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Request__fini(as2_msgs__srv__SetControlMode_Request * msg);

/// Create srv/SetControlMode message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * as2_msgs__srv__SetControlMode_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
as2_msgs__srv__SetControlMode_Request *
as2_msgs__srv__SetControlMode_Request__create();

/// Destroy srv/SetControlMode message.
/**
 * It calls
 * as2_msgs__srv__SetControlMode_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Request__destroy(as2_msgs__srv__SetControlMode_Request * msg);

/// Check for srv/SetControlMode message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Request__are_equal(const as2_msgs__srv__SetControlMode_Request * lhs, const as2_msgs__srv__SetControlMode_Request * rhs);

/// Copy a srv/SetControlMode message.
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
as2_msgs__srv__SetControlMode_Request__copy(
  const as2_msgs__srv__SetControlMode_Request * input,
  as2_msgs__srv__SetControlMode_Request * output);

/// Initialize array of srv/SetControlMode messages.
/**
 * It allocates the memory for the number of elements and calls
 * as2_msgs__srv__SetControlMode_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Request__Sequence__init(as2_msgs__srv__SetControlMode_Request__Sequence * array, size_t size);

/// Finalize array of srv/SetControlMode messages.
/**
 * It calls
 * as2_msgs__srv__SetControlMode_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Request__Sequence__fini(as2_msgs__srv__SetControlMode_Request__Sequence * array);

/// Create array of srv/SetControlMode messages.
/**
 * It allocates the memory for the array and calls
 * as2_msgs__srv__SetControlMode_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
as2_msgs__srv__SetControlMode_Request__Sequence *
as2_msgs__srv__SetControlMode_Request__Sequence__create(size_t size);

/// Destroy array of srv/SetControlMode messages.
/**
 * It calls
 * as2_msgs__srv__SetControlMode_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Request__Sequence__destroy(as2_msgs__srv__SetControlMode_Request__Sequence * array);

/// Check for srv/SetControlMode message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Request__Sequence__are_equal(const as2_msgs__srv__SetControlMode_Request__Sequence * lhs, const as2_msgs__srv__SetControlMode_Request__Sequence * rhs);

/// Copy an array of srv/SetControlMode messages.
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
as2_msgs__srv__SetControlMode_Request__Sequence__copy(
  const as2_msgs__srv__SetControlMode_Request__Sequence * input,
  as2_msgs__srv__SetControlMode_Request__Sequence * output);

/// Initialize srv/SetControlMode message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * as2_msgs__srv__SetControlMode_Response
 * )) before or use
 * as2_msgs__srv__SetControlMode_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Response__init(as2_msgs__srv__SetControlMode_Response * msg);

/// Finalize srv/SetControlMode message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Response__fini(as2_msgs__srv__SetControlMode_Response * msg);

/// Create srv/SetControlMode message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * as2_msgs__srv__SetControlMode_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
as2_msgs__srv__SetControlMode_Response *
as2_msgs__srv__SetControlMode_Response__create();

/// Destroy srv/SetControlMode message.
/**
 * It calls
 * as2_msgs__srv__SetControlMode_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Response__destroy(as2_msgs__srv__SetControlMode_Response * msg);

/// Check for srv/SetControlMode message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Response__are_equal(const as2_msgs__srv__SetControlMode_Response * lhs, const as2_msgs__srv__SetControlMode_Response * rhs);

/// Copy a srv/SetControlMode message.
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
as2_msgs__srv__SetControlMode_Response__copy(
  const as2_msgs__srv__SetControlMode_Response * input,
  as2_msgs__srv__SetControlMode_Response * output);

/// Initialize array of srv/SetControlMode messages.
/**
 * It allocates the memory for the number of elements and calls
 * as2_msgs__srv__SetControlMode_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Response__Sequence__init(as2_msgs__srv__SetControlMode_Response__Sequence * array, size_t size);

/// Finalize array of srv/SetControlMode messages.
/**
 * It calls
 * as2_msgs__srv__SetControlMode_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Response__Sequence__fini(as2_msgs__srv__SetControlMode_Response__Sequence * array);

/// Create array of srv/SetControlMode messages.
/**
 * It allocates the memory for the array and calls
 * as2_msgs__srv__SetControlMode_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
as2_msgs__srv__SetControlMode_Response__Sequence *
as2_msgs__srv__SetControlMode_Response__Sequence__create(size_t size);

/// Destroy array of srv/SetControlMode messages.
/**
 * It calls
 * as2_msgs__srv__SetControlMode_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
void
as2_msgs__srv__SetControlMode_Response__Sequence__destroy(as2_msgs__srv__SetControlMode_Response__Sequence * array);

/// Check for srv/SetControlMode message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_as2_msgs
bool
as2_msgs__srv__SetControlMode_Response__Sequence__are_equal(const as2_msgs__srv__SetControlMode_Response__Sequence * lhs, const as2_msgs__srv__SetControlMode_Response__Sequence * rhs);

/// Copy an array of srv/SetControlMode messages.
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
as2_msgs__srv__SetControlMode_Response__Sequence__copy(
  const as2_msgs__srv__SetControlMode_Response__Sequence * input,
  as2_msgs__srv__SetControlMode_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // AS2_MSGS__SRV__DETAIL__SET_CONTROL_MODE__FUNCTIONS_H_
