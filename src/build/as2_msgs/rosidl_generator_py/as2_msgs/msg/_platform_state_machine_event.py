# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/PlatformStateMachineEvent.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PlatformStateMachineEvent(type):
    """Metaclass of message 'PlatformStateMachineEvent'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'EMERGENCY': -1,
        'ARM': 0,
        'DISARM': 1,
        'TAKE_OFF': 2,
        'TOOK_OFF': 3,
        'LAND': 4,
        'LANDED': 5,
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
                'as2_msgs.msg.PlatformStateMachineEvent')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__platform_state_machine_event
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__platform_state_machine_event
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__platform_state_machine_event
            cls._TYPE_SUPPORT = module.type_support_msg__msg__platform_state_machine_event
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__platform_state_machine_event

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'EMERGENCY': cls.__constants['EMERGENCY'],
            'ARM': cls.__constants['ARM'],
            'DISARM': cls.__constants['DISARM'],
            'TAKE_OFF': cls.__constants['TAKE_OFF'],
            'TOOK_OFF': cls.__constants['TOOK_OFF'],
            'LAND': cls.__constants['LAND'],
            'LANDED': cls.__constants['LANDED'],
        }

    @property
    def EMERGENCY(self):
        """Message constant 'EMERGENCY'."""
        return Metaclass_PlatformStateMachineEvent.__constants['EMERGENCY']

    @property
    def ARM(self):
        """Message constant 'ARM'."""
        return Metaclass_PlatformStateMachineEvent.__constants['ARM']

    @property
    def DISARM(self):
        """Message constant 'DISARM'."""
        return Metaclass_PlatformStateMachineEvent.__constants['DISARM']

    @property
    def TAKE_OFF(self):
        """Message constant 'TAKE_OFF'."""
        return Metaclass_PlatformStateMachineEvent.__constants['TAKE_OFF']

    @property
    def TOOK_OFF(self):
        """Message constant 'TOOK_OFF'."""
        return Metaclass_PlatformStateMachineEvent.__constants['TOOK_OFF']

    @property
    def LAND(self):
        """Message constant 'LAND'."""
        return Metaclass_PlatformStateMachineEvent.__constants['LAND']

    @property
    def LANDED(self):
        """Message constant 'LANDED'."""
        return Metaclass_PlatformStateMachineEvent.__constants['LANDED']


class PlatformStateMachineEvent(metaclass=Metaclass_PlatformStateMachineEvent):
    """
    Message class 'PlatformStateMachineEvent'.

    Constants:
      EMERGENCY
      ARM
      DISARM
      TAKE_OFF
      TOOK_OFF
      LAND
      LANDED
    """

    __slots__ = [
        '_event',
    ]

    _fields_and_field_types = {
        'event': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.event = kwargs.get('event', int())

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
        if self.event != other.event:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def event(self):
        """Message field 'event'."""
        return self._event

    @event.setter
    def event(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'event' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'event' field must be an integer in [-128, 127]"
        self._event = value
