# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/YawMode.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_YawMode(type):
    """Metaclass of message 'YawMode'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'KEEP_YAW': 0,
        'PATH_FACING': 1,
        'FIXED_YAW': 2,
        'YAW_FROM_TOPIC': 3,
        'YAW_FROM_ORIENTATION': 4,
        'YAW_TO_FRAME': 5,
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
                'as2_msgs.msg.YawMode')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__yaw_mode
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__yaw_mode
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__yaw_mode
            cls._TYPE_SUPPORT = module.type_support_msg__msg__yaw_mode
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__yaw_mode

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'KEEP_YAW': cls.__constants['KEEP_YAW'],
            'PATH_FACING': cls.__constants['PATH_FACING'],
            'FIXED_YAW': cls.__constants['FIXED_YAW'],
            'YAW_FROM_TOPIC': cls.__constants['YAW_FROM_TOPIC'],
            'YAW_FROM_ORIENTATION': cls.__constants['YAW_FROM_ORIENTATION'],
            'YAW_TO_FRAME': cls.__constants['YAW_TO_FRAME'],
        }

    @property
    def KEEP_YAW(self):
        """Message constant 'KEEP_YAW'."""
        return Metaclass_YawMode.__constants['KEEP_YAW']

    @property
    def PATH_FACING(self):
        """Message constant 'PATH_FACING'."""
        return Metaclass_YawMode.__constants['PATH_FACING']

    @property
    def FIXED_YAW(self):
        """Message constant 'FIXED_YAW'."""
        return Metaclass_YawMode.__constants['FIXED_YAW']

    @property
    def YAW_FROM_TOPIC(self):
        """Message constant 'YAW_FROM_TOPIC'."""
        return Metaclass_YawMode.__constants['YAW_FROM_TOPIC']

    @property
    def YAW_FROM_ORIENTATION(self):
        """Message constant 'YAW_FROM_ORIENTATION'."""
        return Metaclass_YawMode.__constants['YAW_FROM_ORIENTATION']

    @property
    def YAW_TO_FRAME(self):
        """Message constant 'YAW_TO_FRAME'."""
        return Metaclass_YawMode.__constants['YAW_TO_FRAME']


class YawMode(metaclass=Metaclass_YawMode):
    """
    Message class 'YawMode'.

    Constants:
      KEEP_YAW
      PATH_FACING
      FIXED_YAW
      YAW_FROM_TOPIC
      YAW_FROM_ORIENTATION
      YAW_TO_FRAME
    """

    __slots__ = [
        '_mode',
        '_angle',
    ]

    _fields_and_field_types = {
        'mode': 'uint8',
        'angle': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.mode = kwargs.get('mode', int())
        self.angle = kwargs.get('angle', float())

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
        if self.mode != other.mode:
            return False
        if self.angle != other.angle:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'mode' field must be an unsigned integer in [0, 255]"
        self._mode = value

    @builtins.property
    def angle(self):
        """Message field 'angle'."""
        return self._angle

    @angle.setter
    def angle(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'angle' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'angle' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._angle = value
