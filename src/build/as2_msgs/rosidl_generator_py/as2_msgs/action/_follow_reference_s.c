// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from as2_msgs:action/FollowReference.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "as2_msgs/action/detail/follow_reference__struct.h"
#include "as2_msgs/action/detail/follow_reference__functions.h"

bool as2_msgs__msg__yaw_mode__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * as2_msgs__msg__yaw_mode__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__point_stamped__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__point_stamped__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__goal__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[55];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_Goal", full_classname_dest, 54) == 0);
  }
  as2_msgs__action__FollowReference_Goal * ros_message = _ros_message;
  {  // yaw
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw");
    if (!field) {
      return false;
    }
    if (!as2_msgs__msg__yaw_mode__convert_from_py(field, &ros_message->yaw)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // target_pose
    PyObject * field = PyObject_GetAttrString(_pymsg, "target_pose");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__point_stamped__convert_from_py(field, &ros_message->target_pose)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // max_speed_x
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_speed_x");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_speed_x = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_speed_y
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_speed_y");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_speed_y = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_speed_z
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_speed_z");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_speed_z = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__goal__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_Goal */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_Goal");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_Goal * ros_message = (as2_msgs__action__FollowReference_Goal *)raw_ros_message;
  {  // yaw
    PyObject * field = NULL;
    field = as2_msgs__msg__yaw_mode__convert_to_py(&ros_message->yaw);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // target_pose
    PyObject * field = NULL;
    field = geometry_msgs__msg__point_stamped__convert_to_py(&ros_message->target_pose);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "target_pose", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_speed_x
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_speed_x);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_speed_x", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_speed_y
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_speed_y);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_speed_y", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_speed_z
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_speed_z);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_speed_z", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__result__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[57];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_Result", full_classname_dest, 56) == 0);
  }
  as2_msgs__action__FollowReference_Result * ros_message = _ros_message;
  {  // follow_reference_success
    PyObject * field = PyObject_GetAttrString(_pymsg, "follow_reference_success");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->follow_reference_success = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__result__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_Result */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_Result");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_Result * ros_message = (as2_msgs__action__FollowReference_Result *)raw_ros_message;
  {  // follow_reference_success
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->follow_reference_success ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "follow_reference_success", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__feedback__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[59];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_Feedback", full_classname_dest, 58) == 0);
  }
  as2_msgs__action__FollowReference_Feedback * ros_message = _ros_message;
  {  // actual_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "actual_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->actual_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // actual_distance_to_goal
    PyObject * field = PyObject_GetAttrString(_pymsg, "actual_distance_to_goal");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->actual_distance_to_goal = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__feedback__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_Feedback */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_Feedback");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_Feedback * ros_message = (as2_msgs__action__FollowReference_Feedback *)raw_ros_message;
  {  // actual_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->actual_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "actual_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // actual_distance_to_goal
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->actual_distance_to_goal);
    {
      int rc = PyObject_SetAttrString(_pymessage, "actual_distance_to_goal", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool unique_identifier_msgs__msg__uuid__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * unique_identifier_msgs__msg__uuid__convert_to_py(void * raw_ros_message);
bool as2_msgs__action__follow_reference__goal__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * as2_msgs__action__follow_reference__goal__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__send_goal__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[67];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_SendGoal_Request", full_classname_dest, 66) == 0);
  }
  as2_msgs__action__FollowReference_SendGoal_Request * ros_message = _ros_message;
  {  // goal_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "goal_id");
    if (!field) {
      return false;
    }
    if (!unique_identifier_msgs__msg__uuid__convert_from_py(field, &ros_message->goal_id)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // goal
    PyObject * field = PyObject_GetAttrString(_pymsg, "goal");
    if (!field) {
      return false;
    }
    if (!as2_msgs__action__follow_reference__goal__convert_from_py(field, &ros_message->goal)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__send_goal__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_SendGoal_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_SendGoal_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_SendGoal_Request * ros_message = (as2_msgs__action__FollowReference_SendGoal_Request *)raw_ros_message;
  {  // goal_id
    PyObject * field = NULL;
    field = unique_identifier_msgs__msg__uuid__convert_to_py(&ros_message->goal_id);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "goal_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // goal
    PyObject * field = NULL;
    field = as2_msgs__action__follow_reference__goal__convert_to_py(&ros_message->goal);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "goal", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool builtin_interfaces__msg__time__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * builtin_interfaces__msg__time__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__send_goal__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[68];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_SendGoal_Response", full_classname_dest, 67) == 0);
  }
  as2_msgs__action__FollowReference_SendGoal_Response * ros_message = _ros_message;
  {  // accepted
    PyObject * field = PyObject_GetAttrString(_pymsg, "accepted");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->accepted = (Py_True == field);
    Py_DECREF(field);
  }
  {  // stamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "stamp");
    if (!field) {
      return false;
    }
    if (!builtin_interfaces__msg__time__convert_from_py(field, &ros_message->stamp)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__send_goal__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_SendGoal_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_SendGoal_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_SendGoal_Response * ros_message = (as2_msgs__action__FollowReference_SendGoal_Response *)raw_ros_message;
  {  // accepted
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->accepted ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "accepted", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // stamp
    PyObject * field = NULL;
    field = builtin_interfaces__msg__time__convert_to_py(&ros_message->stamp);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "stamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool unique_identifier_msgs__msg__uuid__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * unique_identifier_msgs__msg__uuid__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__get_result__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[68];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_GetResult_Request", full_classname_dest, 67) == 0);
  }
  as2_msgs__action__FollowReference_GetResult_Request * ros_message = _ros_message;
  {  // goal_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "goal_id");
    if (!field) {
      return false;
    }
    if (!unique_identifier_msgs__msg__uuid__convert_from_py(field, &ros_message->goal_id)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__get_result__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_GetResult_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_GetResult_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_GetResult_Request * ros_message = (as2_msgs__action__FollowReference_GetResult_Request *)raw_ros_message;
  {  // goal_id
    PyObject * field = NULL;
    field = unique_identifier_msgs__msg__uuid__convert_to_py(&ros_message->goal_id);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "goal_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"

bool as2_msgs__action__follow_reference__result__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * as2_msgs__action__follow_reference__result__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__get_result__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[69];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_GetResult_Response", full_classname_dest, 68) == 0);
  }
  as2_msgs__action__FollowReference_GetResult_Response * ros_message = _ros_message;
  {  // status
    PyObject * field = PyObject_GetAttrString(_pymsg, "status");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->status = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // result
    PyObject * field = PyObject_GetAttrString(_pymsg, "result");
    if (!field) {
      return false;
    }
    if (!as2_msgs__action__follow_reference__result__convert_from_py(field, &ros_message->result)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__get_result__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_GetResult_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_GetResult_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_GetResult_Response * ros_message = (as2_msgs__action__FollowReference_GetResult_Response *)raw_ros_message;
  {  // status
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->status);
    {
      int rc = PyObject_SetAttrString(_pymessage, "status", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // result
    PyObject * field = NULL;
    field = as2_msgs__action__follow_reference__result__convert_to_py(&ros_message->result);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "result", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__struct.h"
// already included above
// #include "as2_msgs/action/detail/follow_reference__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool unique_identifier_msgs__msg__uuid__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * unique_identifier_msgs__msg__uuid__convert_to_py(void * raw_ros_message);
bool as2_msgs__action__follow_reference__feedback__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * as2_msgs__action__follow_reference__feedback__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__action__follow_reference__feedback_message__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[66];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("as2_msgs.action._follow_reference.FollowReference_FeedbackMessage", full_classname_dest, 65) == 0);
  }
  as2_msgs__action__FollowReference_FeedbackMessage * ros_message = _ros_message;
  {  // goal_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "goal_id");
    if (!field) {
      return false;
    }
    if (!unique_identifier_msgs__msg__uuid__convert_from_py(field, &ros_message->goal_id)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // feedback
    PyObject * field = PyObject_GetAttrString(_pymsg, "feedback");
    if (!field) {
      return false;
    }
    if (!as2_msgs__action__follow_reference__feedback__convert_from_py(field, &ros_message->feedback)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__action__follow_reference__feedback_message__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of FollowReference_FeedbackMessage */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.action._follow_reference");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "FollowReference_FeedbackMessage");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__action__FollowReference_FeedbackMessage * ros_message = (as2_msgs__action__FollowReference_FeedbackMessage *)raw_ros_message;
  {  // goal_id
    PyObject * field = NULL;
    field = unique_identifier_msgs__msg__uuid__convert_to_py(&ros_message->goal_id);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "goal_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // feedback
    PyObject * field = NULL;
    field = as2_msgs__action__follow_reference__feedback__convert_to_py(&ros_message->feedback);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "feedback", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
