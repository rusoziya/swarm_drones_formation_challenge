# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:srv/GeopathToPath.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GeopathToPath_Request(type):
    """Metaclass of message 'GeopathToPath_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('as2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'as2_msgs.srv.GeopathToPath_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__geopath_to_path__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__geopath_to_path__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__geopath_to_path__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__geopath_to_path__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__geopath_to_path__request

            from geographic_msgs.msg import GeoPath
            if GeoPath.__class__._TYPE_SUPPORT is None:
                GeoPath.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class GeopathToPath_Request(metaclass=Metaclass_GeopathToPath_Request):
    """Message class 'GeopathToPath_Request'."""

    __slots__ = [
        '_geo_path',
    ]

    _fields_and_field_types = {
        'geo_path': 'geographic_msgs/GeoPath',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['geographic_msgs', 'msg'], 'GeoPath'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from geographic_msgs.msg import GeoPath
        self.geo_path = kwargs.get('geo_path', GeoPath())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.geo_path != other.geo_path:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def geo_path(self):
        """Message field 'geo_path'."""
        return self._geo_path

    @geo_path.setter
    def geo_path(self, value):
        if __debug__:
            from geographic_msgs.msg import GeoPath
            assert \
                isinstance(value, GeoPath), \
                "The 'geo_path' field must be a sub message of type 'GeoPath'"
        self._geo_path = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_GeopathToPath_Response(type):
    """Metaclass of message 'GeopathToPath_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('as2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'as2_msgs.srv.GeopathToPath_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__geopath_to_path__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__geopath_to_path__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__geopath_to_path__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__geopath_to_path__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__geopath_to_path__response

            from nav_msgs.msg import Path
            if Path.__class__._TYPE_SUPPORT is None:
                Path.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class GeopathToPath_Response(metaclass=Metaclass_GeopathToPath_Response):
    """Message class 'GeopathToPath_Response'."""

    __slots__ = [
        '_success',
        '_path',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
        'path': 'nav_msgs/Path',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['nav_msgs', 'msg'], 'Path'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())
        from nav_msgs.msg import Path
        self.path = kwargs.get('path', Path())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.success != other.success:
            return False
        if self.path != other.path:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value

    @builtins.property
    def path(self):
        """Message field 'path'."""
        return self._path

    @path.setter
    def path(self, value):
        if __debug__:
            from nav_msgs.msg import Path
            assert \
                isinstance(value, Path), \
                "The 'path' field must be a sub message of type 'Path'"
        self._path = value


class Metaclass_GeopathToPath(type):
    """Metaclass of service 'GeopathToPath'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('as2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'as2_msgs.srv.GeopathToPath')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__geopath_to_path

            from as2_msgs.srv import _geopath_to_path
            if _geopath_to_path.Metaclass_GeopathToPath_Request._TYPE_SUPPORT is None:
                _geopath_to_path.Metaclass_GeopathToPath_Request.__import_type_support__()
            if _geopath_to_path.Metaclass_GeopathToPath_Response._TYPE_SUPPORT is None:
                _geopath_to_path.Metaclass_GeopathToPath_Response.__import_type_support__()


class GeopathToPath(metaclass=Metaclass_GeopathToPath):
    from as2_msgs.srv._geopath_to_path import GeopathToPath_Request as Request
    from as2_msgs.srv._geopath_to_path import GeopathToPath_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
