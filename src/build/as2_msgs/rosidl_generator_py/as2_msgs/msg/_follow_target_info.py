# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/FollowTargetInfo.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_FollowTargetInfo(type):
    """Metaclass of message 'FollowTargetInfo'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'WAITING': 0,
        'RUNNING': 1,
        'END': 2,
        'UNSET': 0,
        'PICKUP': 1,
        'UNPICK': 2,
        'DYNAMIC_LAND': 3,
        'DYNAMIC_FOLLOWER': 4,
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
                'as2_msgs.msg.FollowTargetInfo')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__follow_target_info
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__follow_target_info
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__follow_target_info
            cls._TYPE_SUPPORT = module.type_support_msg__msg__follow_target_info
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__follow_target_info

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'WAITING': cls.__constants['WAITING'],
            'RUNNING': cls.__constants['RUNNING'],
            'END': cls.__constants['END'],
            'UNSET': cls.__constants['UNSET'],
            'PICKUP': cls.__constants['PICKUP'],
            'UNPICK': cls.__constants['UNPICK'],
            'DYNAMIC_LAND': cls.__constants['DYNAMIC_LAND'],
            'DYNAMIC_FOLLOWER': cls.__constants['DYNAMIC_FOLLOWER'],
        }

    @property
    def WAITING(self):
        """Message constant 'WAITING'."""
        return Metaclass_FollowTargetInfo.__constants['WAITING']

    @property
    def RUNNING(self):
        """Message constant 'RUNNING'."""
        return Metaclass_FollowTargetInfo.__constants['RUNNING']

    @property
    def END(self):
        """Message constant 'END'."""
        return Metaclass_FollowTargetInfo.__constants['END']

    @property
    def UNSET(self):
        """Message constant 'UNSET'."""
        return Metaclass_FollowTargetInfo.__constants['UNSET']

    @property
    def PICKUP(self):
        """Message constant 'PICKUP'."""
        return Metaclass_FollowTargetInfo.__constants['PICKUP']

    @property
    def UNPICK(self):
        """Message constant 'UNPICK'."""
        return Metaclass_FollowTargetInfo.__constants['UNPICK']

    @property
    def DYNAMIC_LAND(self):
        """Message constant 'DYNAMIC_LAND'."""
        return Metaclass_FollowTargetInfo.__constants['DYNAMIC_LAND']

    @property
    def DYNAMIC_FOLLOWER(self):
        """Message constant 'DYNAMIC_FOLLOWER'."""
        return Metaclass_FollowTargetInfo.__constants['DYNAMIC_FOLLOWER']


class FollowTargetInfo(metaclass=Metaclass_FollowTargetInfo):
    """
    Message class 'FollowTargetInfo'.

    Constants:
      WAITING
      RUNNING
      END
      UNSET
      PICKUP
      UNPICK
      DYNAMIC_LAND
      DYNAMIC_FOLLOWER
    """

    __slots__ = [
        '_header',
        '_follow_status',
        '_follow_mode',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'follow_status': 'int8',
        'follow_mode': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.follow_status = kwargs.get('follow_status', int())
        self.follow_mode = kwargs.get('follow_mode', int())

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
        if self.header != other.header:
            return False
        if self.follow_status != other.follow_status:
            return False
        if self.follow_mode != other.follow_mode:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def follow_status(self):
        """Message field 'follow_status'."""
        return self._follow_status

    @follow_status.setter
    def follow_status(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'follow_status' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'follow_status' field must be an integer in [-128, 127]"
        self._follow_status = value

    @builtins.property
    def follow_mode(self):
        """Message field 'follow_mode'."""
        return self._follow_mode

    @follow_mode.setter
    def follow_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'follow_mode' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'follow_mode' field must be an integer in [-128, 127]"
        self._follow_mode = value
