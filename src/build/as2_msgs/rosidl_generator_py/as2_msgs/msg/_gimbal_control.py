# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/GimbalControl.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GimbalControl(type):
    """Metaclass of message 'GimbalControl'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'POSITION_MODE': 0,
        'SPEED_MODE': 1,
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
                'as2_msgs.msg.GimbalControl')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__gimbal_control
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__gimbal_control
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__gimbal_control
            cls._TYPE_SUPPORT = module.type_support_msg__msg__gimbal_control
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__gimbal_control

            from geometry_msgs.msg import Vector3Stamped
            if Vector3Stamped.__class__._TYPE_SUPPORT is None:
                Vector3Stamped.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'POSITION_MODE': cls.__constants['POSITION_MODE'],
            'SPEED_MODE': cls.__constants['SPEED_MODE'],
        }

    @property
    def POSITION_MODE(self):
        """Message constant 'POSITION_MODE'."""
        return Metaclass_GimbalControl.__constants['POSITION_MODE']

    @property
    def SPEED_MODE(self):
        """Message constant 'SPEED_MODE'."""
        return Metaclass_GimbalControl.__constants['SPEED_MODE']


class GimbalControl(metaclass=Metaclass_GimbalControl):
    """
    Message class 'GimbalControl'.

    Constants:
      POSITION_MODE
      SPEED_MODE
    """

    __slots__ = [
        '_control_mode',
        '_target',
    ]

    _fields_and_field_types = {
        'control_mode': 'uint8',
        'target': 'geometry_msgs/Vector3Stamped',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3Stamped'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.control_mode = kwargs.get('control_mode', int())
        from geometry_msgs.msg import Vector3Stamped
        self.target = kwargs.get('target', Vector3Stamped())

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
        if self.control_mode != other.control_mode:
            return False
        if self.target != other.target:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def control_mode(self):
        """Message field 'control_mode'."""
        return self._control_mode

    @control_mode.setter
    def control_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'control_mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'control_mode' field must be an unsigned integer in [0, 255]"
        self._control_mode = value

    @builtins.property
    def target(self):
        """Message field 'target'."""
        return self._target

    @target.setter
    def target(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3Stamped
            assert \
                isinstance(value, Vector3Stamped), \
                "The 'target' field must be a sub message of type 'Vector3Stamped'"
        self._target = value
