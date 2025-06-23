# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/AlertEvent.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_AlertEvent(type):
    """Metaclass of message 'AlertEvent'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'KILL_SWITCH': -1,
        'EMERGENCY_HOVER': -2,
        'EMERGENCY_LAND': -3,
        'INFO_ALERT': 0,
        'FORCE_HOVER': 1,
        'FORCE_LAND': 2,
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
                'as2_msgs.msg.AlertEvent')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__alert_event
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__alert_event
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__alert_event
            cls._TYPE_SUPPORT = module.type_support_msg__msg__alert_event
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__alert_event

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'KILL_SWITCH': cls.__constants['KILL_SWITCH'],
            'EMERGENCY_HOVER': cls.__constants['EMERGENCY_HOVER'],
            'EMERGENCY_LAND': cls.__constants['EMERGENCY_LAND'],
            'INFO_ALERT': cls.__constants['INFO_ALERT'],
            'FORCE_HOVER': cls.__constants['FORCE_HOVER'],
            'FORCE_LAND': cls.__constants['FORCE_LAND'],
        }

    @property
    def KILL_SWITCH(self):
        """Message constant 'KILL_SWITCH'."""
        return Metaclass_AlertEvent.__constants['KILL_SWITCH']

    @property
    def EMERGENCY_HOVER(self):
        """Message constant 'EMERGENCY_HOVER'."""
        return Metaclass_AlertEvent.__constants['EMERGENCY_HOVER']

    @property
    def EMERGENCY_LAND(self):
        """Message constant 'EMERGENCY_LAND'."""
        return Metaclass_AlertEvent.__constants['EMERGENCY_LAND']

    @property
    def INFO_ALERT(self):
        """Message constant 'INFO_ALERT'."""
        return Metaclass_AlertEvent.__constants['INFO_ALERT']

    @property
    def FORCE_HOVER(self):
        """Message constant 'FORCE_HOVER'."""
        return Metaclass_AlertEvent.__constants['FORCE_HOVER']

    @property
    def FORCE_LAND(self):
        """Message constant 'FORCE_LAND'."""
        return Metaclass_AlertEvent.__constants['FORCE_LAND']


class AlertEvent(metaclass=Metaclass_AlertEvent):
    """
    Message class 'AlertEvent'.

    Constants:
      KILL_SWITCH
      EMERGENCY_HOVER
      EMERGENCY_LAND
      INFO_ALERT
      FORCE_HOVER
      FORCE_LAND
    """

    __slots__ = [
        '_alert',
        '_description',
    ]

    _fields_and_field_types = {
        'alert': 'int8',
        'description': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.alert = kwargs.get('alert', int())
        self.description = kwargs.get('description', str())

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
        if self.alert != other.alert:
            return False
        if self.description != other.description:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def alert(self):
        """Message field 'alert'."""
        return self._alert

    @alert.setter
    def alert(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'alert' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'alert' field must be an integer in [-128, 127]"
        self._alert = value

    @builtins.property
    def description(self):
        """Message field 'description'."""
        return self._description

    @description.setter
    def description(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'description' field must be of type 'str'"
        self._description = value
