# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:srv/ListControlModes.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ListControlModes_Request(type):
    """Metaclass of message 'ListControlModes_Request'."""

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
                'as2_msgs.srv.ListControlModes_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__list_control_modes__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__list_control_modes__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__list_control_modes__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__list_control_modes__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__list_control_modes__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ListControlModes_Request(metaclass=Metaclass_ListControlModes_Request):
    """Message class 'ListControlModes_Request'."""

    __slots__ = [
    ]

    _fields_and_field_types = {
    }

    SLOT_TYPES = (
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))

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
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)


# Import statements for member types

# Member 'control_modes'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

# already imported above
# import rosidl_parser.definition


class Metaclass_ListControlModes_Response(type):
    """Metaclass of message 'ListControlModes_Response'."""

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
                'as2_msgs.srv.ListControlModes_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__list_control_modes__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__list_control_modes__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__list_control_modes__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__list_control_modes__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__list_control_modes__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ListControlModes_Response(metaclass=Metaclass_ListControlModes_Response):
    """Message class 'ListControlModes_Response'."""

    __slots__ = [
        '_source',
        '_control_modes',
    ]

    _fields_and_field_types = {
        'source': 'string',
        'control_modes': 'sequence<uint8>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.source = kwargs.get('source', str())
        self.control_modes = array.array('B', kwargs.get('control_modes', []))

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
        if self.source != other.source:
            return False
        if self.control_modes != other.control_modes:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def source(self):
        """Message field 'source'."""
        return self._source

    @source.setter
    def source(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'source' field must be of type 'str'"
        self._source = value

    @builtins.property
    def control_modes(self):
        """Message field 'control_modes'."""
        return self._control_modes

    @control_modes.setter
    def control_modes(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'control_modes' array.array() must have the type code of 'B'"
            self._control_modes = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'control_modes' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._control_modes = array.array('B', value)


class Metaclass_ListControlModes(type):
    """Metaclass of service 'ListControlModes'."""

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
                'as2_msgs.srv.ListControlModes')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__list_control_modes

            from as2_msgs.srv import _list_control_modes
            if _list_control_modes.Metaclass_ListControlModes_Request._TYPE_SUPPORT is None:
                _list_control_modes.Metaclass_ListControlModes_Request.__import_type_support__()
            if _list_control_modes.Metaclass_ListControlModes_Response._TYPE_SUPPORT is None:
                _list_control_modes.Metaclass_ListControlModes_Response.__import_type_support__()


class ListControlModes(metaclass=Metaclass_ListControlModes):
    from as2_msgs.srv._list_control_modes import ListControlModes_Request as Request
    from as2_msgs.srv._list_control_modes import ListControlModes_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
