# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:srv/SetGeozone.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetGeozone_Request(type):
    """Metaclass of message 'SetGeozone_Request'."""

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
                'as2_msgs.srv.SetGeozone_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_geozone__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_geozone__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_geozone__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_geozone__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_geozone__request

            from as2_msgs.msg import Geozone
            if Geozone.__class__._TYPE_SUPPORT is None:
                Geozone.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetGeozone_Request(metaclass=Metaclass_SetGeozone_Request):
    """Message class 'SetGeozone_Request'."""

    __slots__ = [
        '_geozone',
    ]

    _fields_and_field_types = {
        'geozone': 'as2_msgs/Geozone',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['as2_msgs', 'msg'], 'Geozone'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from as2_msgs.msg import Geozone
        self.geozone = kwargs.get('geozone', Geozone())

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
        if self.geozone != other.geozone:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def geozone(self):
        """Message field 'geozone'."""
        return self._geozone

    @geozone.setter
    def geozone(self, value):
        if __debug__:
            from as2_msgs.msg import Geozone
            assert \
                isinstance(value, Geozone), \
                "The 'geozone' field must be a sub message of type 'Geozone'"
        self._geozone = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_SetGeozone_Response(type):
    """Metaclass of message 'SetGeozone_Response'."""

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
                'as2_msgs.srv.SetGeozone_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_geozone__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_geozone__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_geozone__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_geozone__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_geozone__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetGeozone_Response(metaclass=Metaclass_SetGeozone_Response):
    """Message class 'SetGeozone_Response'."""

    __slots__ = [
        '_success',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())

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


class Metaclass_SetGeozone(type):
    """Metaclass of service 'SetGeozone'."""

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
                'as2_msgs.srv.SetGeozone')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_geozone

            from as2_msgs.srv import _set_geozone
            if _set_geozone.Metaclass_SetGeozone_Request._TYPE_SUPPORT is None:
                _set_geozone.Metaclass_SetGeozone_Request.__import_type_support__()
            if _set_geozone.Metaclass_SetGeozone_Response._TYPE_SUPPORT is None:
                _set_geozone.Metaclass_SetGeozone_Response.__import_type_support__()


class SetGeozone(metaclass=Metaclass_SetGeozone):
    from as2_msgs.srv._set_geozone import SetGeozone_Request as Request
    from as2_msgs.srv._set_geozone import SetGeozone_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
