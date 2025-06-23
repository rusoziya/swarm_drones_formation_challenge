# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/ControlMode.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ControlMode(type):
    """Metaclass of message 'ControlMode'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'NONE': 0,
        'YAW_ANGLE': 1,
        'YAW_SPEED': 2,
        'UNSET': 0,
        'HOVER': 1,
        'POSITION': 2,
        'SPEED': 3,
        'SPEED_IN_A_PLANE': 4,
        'ATTITUDE': 5,
        'ACRO': 6,
        'TRAJECTORY': 7,
        'UNDEFINED_FRAME': 0,
        'LOCAL_ENU_FRAME': 1,
        'BODY_FLU_FRAME': 2,
        'GLOBAL_LAT_LONG_ASML': 3,
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
                'as2_msgs.msg.ControlMode')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__control_mode
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__control_mode
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__control_mode
            cls._TYPE_SUPPORT = module.type_support_msg__msg__control_mode
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__control_mode

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'NONE': cls.__constants['NONE'],
            'YAW_ANGLE': cls.__constants['YAW_ANGLE'],
            'YAW_SPEED': cls.__constants['YAW_SPEED'],
            'UNSET': cls.__constants['UNSET'],
            'HOVER': cls.__constants['HOVER'],
            'POSITION': cls.__constants['POSITION'],
            'SPEED': cls.__constants['SPEED'],
            'SPEED_IN_A_PLANE': cls.__constants['SPEED_IN_A_PLANE'],
            'ATTITUDE': cls.__constants['ATTITUDE'],
            'ACRO': cls.__constants['ACRO'],
            'TRAJECTORY': cls.__constants['TRAJECTORY'],
            'UNDEFINED_FRAME': cls.__constants['UNDEFINED_FRAME'],
            'LOCAL_ENU_FRAME': cls.__constants['LOCAL_ENU_FRAME'],
            'BODY_FLU_FRAME': cls.__constants['BODY_FLU_FRAME'],
            'GLOBAL_LAT_LONG_ASML': cls.__constants['GLOBAL_LAT_LONG_ASML'],
        }

    @property
    def NONE(self):
        """Message constant 'NONE'."""
        return Metaclass_ControlMode.__constants['NONE']

    @property
    def YAW_ANGLE(self):
        """Message constant 'YAW_ANGLE'."""
        return Metaclass_ControlMode.__constants['YAW_ANGLE']

    @property
    def YAW_SPEED(self):
        """Message constant 'YAW_SPEED'."""
        return Metaclass_ControlMode.__constants['YAW_SPEED']

    @property
    def UNSET(self):
        """Message constant 'UNSET'."""
        return Metaclass_ControlMode.__constants['UNSET']

    @property
    def HOVER(self):
        """Message constant 'HOVER'."""
        return Metaclass_ControlMode.__constants['HOVER']

    @property
    def POSITION(self):
        """Message constant 'POSITION'."""
        return Metaclass_ControlMode.__constants['POSITION']

    @property
    def SPEED(self):
        """Message constant 'SPEED'."""
        return Metaclass_ControlMode.__constants['SPEED']

    @property
    def SPEED_IN_A_PLANE(self):
        """Message constant 'SPEED_IN_A_PLANE'."""
        return Metaclass_ControlMode.__constants['SPEED_IN_A_PLANE']

    @property
    def ATTITUDE(self):
        """Message constant 'ATTITUDE'."""
        return Metaclass_ControlMode.__constants['ATTITUDE']

    @property
    def ACRO(self):
        """Message constant 'ACRO'."""
        return Metaclass_ControlMode.__constants['ACRO']

    @property
    def TRAJECTORY(self):
        """Message constant 'TRAJECTORY'."""
        return Metaclass_ControlMode.__constants['TRAJECTORY']

    @property
    def UNDEFINED_FRAME(self):
        """Message constant 'UNDEFINED_FRAME'."""
        return Metaclass_ControlMode.__constants['UNDEFINED_FRAME']

    @property
    def LOCAL_ENU_FRAME(self):
        """Message constant 'LOCAL_ENU_FRAME'."""
        return Metaclass_ControlMode.__constants['LOCAL_ENU_FRAME']

    @property
    def BODY_FLU_FRAME(self):
        """Message constant 'BODY_FLU_FRAME'."""
        return Metaclass_ControlMode.__constants['BODY_FLU_FRAME']

    @property
    def GLOBAL_LAT_LONG_ASML(self):
        """Message constant 'GLOBAL_LAT_LONG_ASML'."""
        return Metaclass_ControlMode.__constants['GLOBAL_LAT_LONG_ASML']


class ControlMode(metaclass=Metaclass_ControlMode):
    """
    Message class 'ControlMode'.

    Constants:
      NONE
      YAW_ANGLE
      YAW_SPEED
      UNSET
      HOVER
      POSITION
      SPEED
      SPEED_IN_A_PLANE
      ATTITUDE
      ACRO
      TRAJECTORY
      UNDEFINED_FRAME
      LOCAL_ENU_FRAME
      BODY_FLU_FRAME
      GLOBAL_LAT_LONG_ASML
    """

    __slots__ = [
        '_header',
        '_yaw_mode',
        '_control_mode',
        '_reference_frame',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'yaw_mode': 'int8',
        'control_mode': 'int8',
        'reference_frame': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.yaw_mode = kwargs.get('yaw_mode', int())
        self.control_mode = kwargs.get('control_mode', int())
        self.reference_frame = kwargs.get('reference_frame', int())

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
        if self.yaw_mode != other.yaw_mode:
            return False
        if self.control_mode != other.control_mode:
            return False
        if self.reference_frame != other.reference_frame:
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
    def yaw_mode(self):
        """Message field 'yaw_mode'."""
        return self._yaw_mode

    @yaw_mode.setter
    def yaw_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'yaw_mode' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'yaw_mode' field must be an integer in [-128, 127]"
        self._yaw_mode = value

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
            assert value >= -128 and value < 128, \
                "The 'control_mode' field must be an integer in [-128, 127]"
        self._control_mode = value

    @builtins.property
    def reference_frame(self):
        """Message field 'reference_frame'."""
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'reference_frame' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'reference_frame' field must be an integer in [-128, 127]"
        self._reference_frame = value
