# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/PlatformStatus.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PlatformStatus(type):
    """Metaclass of message 'PlatformStatus'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'EMERGENCY': -1,
        'DISARMED': 0,
        'LANDED': 1,
        'TAKING_OFF': 2,
        'FLYING': 3,
        'LANDING': 4,
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
                'as2_msgs.msg.PlatformStatus')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__platform_status
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__platform_status
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__platform_status
            cls._TYPE_SUPPORT = module.type_support_msg__msg__platform_status
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__platform_status

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'EMERGENCY': cls.__constants['EMERGENCY'],
            'DISARMED': cls.__constants['DISARMED'],
            'LANDED': cls.__constants['LANDED'],
            'TAKING_OFF': cls.__constants['TAKING_OFF'],
            'FLYING': cls.__constants['FLYING'],
            'LANDING': cls.__constants['LANDING'],
        }

    @property
    def EMERGENCY(self):
        """Message constant 'EMERGENCY'."""
        return Metaclass_PlatformStatus.__constants['EMERGENCY']

    @property
    def DISARMED(self):
        """Message constant 'DISARMED'."""
        return Metaclass_PlatformStatus.__constants['DISARMED']

    @property
    def LANDED(self):
        """Message constant 'LANDED'."""
        return Metaclass_PlatformStatus.__constants['LANDED']

    @property
    def TAKING_OFF(self):
        """Message constant 'TAKING_OFF'."""
        return Metaclass_PlatformStatus.__constants['TAKING_OFF']

    @property
    def FLYING(self):
        """Message constant 'FLYING'."""
        return Metaclass_PlatformStatus.__constants['FLYING']

    @property
    def LANDING(self):
        """Message constant 'LANDING'."""
        return Metaclass_PlatformStatus.__constants['LANDING']


class PlatformStatus(metaclass=Metaclass_PlatformStatus):
    """
    Message class 'PlatformStatus'.

    Constants:
      EMERGENCY
      DISARMED
      LANDED
      TAKING_OFF
      FLYING
      LANDING
    """

    __slots__ = [
        '_state',
    ]

    _fields_and_field_types = {
        'state': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.state = kwargs.get('state', int())

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
        if self.state != other.state:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def state(self):
        """Message field 'state'."""
        return self._state

    @state.setter
    def state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'state' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'state' field must be an integer in [-128, 127]"
        self._state = value
