// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from as2_msgs:srv/PackageUnPick.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "as2_msgs/srv/detail/package_un_pick__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace as2_msgs
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PackageUnPick_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PackageUnPick_Request_type_support_ids_t;

static const _PackageUnPick_Request_type_support_ids_t _PackageUnPick_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PackageUnPick_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PackageUnPick_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PackageUnPick_Request_type_support_symbol_names_t _PackageUnPick_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, as2_msgs, srv, PackageUnPick_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, srv, PackageUnPick_Request)),
  }
};

typedef struct _PackageUnPick_Request_type_support_data_t
{
  void * data[2];
} _PackageUnPick_Request_type_support_data_t;

static _PackageUnPick_Request_type_support_data_t _PackageUnPick_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PackageUnPick_Request_message_typesupport_map = {
  2,
  "as2_msgs",
  &_PackageUnPick_Request_message_typesupport_ids.typesupport_identifier[0],
  &_PackageUnPick_Request_message_typesupport_symbol_names.symbol_name[0],
  &_PackageUnPick_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t PackageUnPick_Request_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PackageUnPick_Request_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace as2_msgs

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::srv::PackageUnPick_Request>()
{
  return &::as2_msgs::srv::rosidl_typesupport_cpp::PackageUnPick_Request_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, as2_msgs, srv, PackageUnPick_Request)() {
  return get_message_type_support_handle<as2_msgs::srv::PackageUnPick_Request>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "as2_msgs/srv/detail/package_un_pick__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace as2_msgs
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PackageUnPick_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PackageUnPick_Response_type_support_ids_t;

static const _PackageUnPick_Response_type_support_ids_t _PackageUnPick_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PackageUnPick_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PackageUnPick_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PackageUnPick_Response_type_support_symbol_names_t _PackageUnPick_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, as2_msgs, srv, PackageUnPick_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, srv, PackageUnPick_Response)),
  }
};

typedef struct _PackageUnPick_Response_type_support_data_t
{
  void * data[2];
} _PackageUnPick_Response_type_support_data_t;

static _PackageUnPick_Response_type_support_data_t _PackageUnPick_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PackageUnPick_Response_message_typesupport_map = {
  2,
  "as2_msgs",
  &_PackageUnPick_Response_message_typesupport_ids.typesupport_identifier[0],
  &_PackageUnPick_Response_message_typesupport_symbol_names.symbol_name[0],
  &_PackageUnPick_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t PackageUnPick_Response_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PackageUnPick_Response_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace as2_msgs

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<as2_msgs::srv::PackageUnPick_Response>()
{
  return &::as2_msgs::srv::rosidl_typesupport_cpp::PackageUnPick_Response_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, as2_msgs, srv, PackageUnPick_Response)() {
  return get_message_type_support_handle<as2_msgs::srv::PackageUnPick_Response>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "as2_msgs/srv/detail/package_un_pick__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/service_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace as2_msgs
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _PackageUnPick_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _PackageUnPick_type_support_ids_t;

static const _PackageUnPick_type_support_ids_t _PackageUnPick_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _PackageUnPick_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _PackageUnPick_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _PackageUnPick_type_support_symbol_names_t _PackageUnPick_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, as2_msgs, srv, PackageUnPick)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, as2_msgs, srv, PackageUnPick)),
  }
};

typedef struct _PackageUnPick_type_support_data_t
{
  void * data[2];
} _PackageUnPick_type_support_data_t;

static _PackageUnPick_type_support_data_t _PackageUnPick_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _PackageUnPick_service_typesupport_map = {
  2,
  "as2_msgs",
  &_PackageUnPick_service_typesupport_ids.typesupport_identifier[0],
  &_PackageUnPick_service_typesupport_symbol_names.symbol_name[0],
  &_PackageUnPick_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t PackageUnPick_service_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_PackageUnPick_service_typesupport_map),
  ::rosidl_typesupport_cpp::get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace as2_msgs

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<as2_msgs::srv::PackageUnPick>()
{
  return &::as2_msgs::srv::rosidl_typesupport_cpp::PackageUnPick_service_type_support_handle;
}

}  // namespace rosidl_typesupport_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_cpp, as2_msgs, srv, PackageUnPick)() {
  return ::rosidl_typesupport_cpp::get_service_type_support_handle<as2_msgs::srv::PackageUnPick>();
}

#ifdef __cplusplus
}
#endif
