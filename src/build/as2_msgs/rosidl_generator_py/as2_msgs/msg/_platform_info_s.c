// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from as2_msgs:msg/PlatformInfo.idl
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
#include "as2_msgs/msg/detail/platform_info__struct.h"
#include "as2_msgs/msg/detail/platform_info__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);
bool as2_msgs__msg__platform_status__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * as2_msgs__msg__platform_status__convert_to_py(void * raw_ros_message);
bool as2_msgs__msg__control_mode__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * as2_msgs__msg__control_mode__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool as2_msgs__msg__platform_info__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[41];
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
    assert(strncmp("as2_msgs.msg._platform_info.PlatformInfo", full_classname_dest, 40) == 0);
  }
  as2_msgs__msg__PlatformInfo * ros_message = _ros_message;
  {  // header
    PyObject * field = PyObject_GetAttrString(_pymsg, "header");
    if (!field) {
      return false;
    }
    if (!std_msgs__msg__header__convert_from_py(field, &ros_message->header)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // connected
    PyObject * field = PyObject_GetAttrString(_pymsg, "connected");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->connected = (Py_True == field);
    Py_DECREF(field);
  }
  {  // armed
    PyObject * field = PyObject_GetAttrString(_pymsg, "armed");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->armed = (Py_True == field);
    Py_DECREF(field);
  }
  {  // offboard
    PyObject * field = PyObject_GetAttrString(_pymsg, "offboard");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->offboard = (Py_True == field);
    Py_DECREF(field);
  }
  {  // status
    PyObject * field = PyObject_GetAttrString(_pymsg, "status");
    if (!field) {
      return false;
    }
    if (!as2_msgs__msg__platform_status__convert_from_py(field, &ros_message->status)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // current_control_mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "current_control_mode");
    if (!field) {
      return false;
    }
    if (!as2_msgs__msg__control_mode__convert_from_py(field, &ros_message->current_control_mode)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * as2_msgs__msg__platform_info__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PlatformInfo */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("as2_msgs.msg._platform_info");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PlatformInfo");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  as2_msgs__msg__PlatformInfo * ros_message = (as2_msgs__msg__PlatformInfo *)raw_ros_message;
  {  // header
    PyObject * field = NULL;
    field = std_msgs__msg__header__convert_to_py(&ros_message->header);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "header", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // connected
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->connected ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "connected", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // armed
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->armed ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "armed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // offboard
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->offboard ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "offboard", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status
    PyObject * field = NULL;
    field = as2_msgs__msg__platform_status__convert_to_py(&ros_message->status);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "status", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // current_control_mode
    PyObject * field = NULL;
    field = as2_msgs__msg__control_mode__convert_to_py(&ros_message->current_control_mode);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "current_control_mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
