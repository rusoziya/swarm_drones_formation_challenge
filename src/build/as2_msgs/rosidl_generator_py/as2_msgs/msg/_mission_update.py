# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/MissionUpdate.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MissionUpdate(type):
    """Metaclass of message 'MissionUpdate'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'EXECUTE': 0,
        'LOAD': 1,
        'START': 2,
        'PAUSE': 3,
        'RESUME': 4,
        'STOP': 5,
        'JUMP_TO': 6,
        'REPEAT': 7,
        'INSERT': 8,
        'MODIFY': 9,
        'REMOVE': 10,
        'RESET': 11,
        'ABORT': 12,
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
                'as2_msgs.msg.MissionUpdate')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__mission_update
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__mission_update
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__mission_update
            cls._TYPE_SUPPORT = module.type_support_msg__msg__mission_update
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__mission_update

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'EXECUTE': cls.__constants['EXECUTE'],
            'LOAD': cls.__constants['LOAD'],
            'START': cls.__constants['START'],
            'PAUSE': cls.__constants['PAUSE'],
            'RESUME': cls.__constants['RESUME'],
            'STOP': cls.__constants['STOP'],
            'JUMP_TO': cls.__constants['JUMP_TO'],
            'REPEAT': cls.__constants['REPEAT'],
            'INSERT': cls.__constants['INSERT'],
            'MODIFY': cls.__constants['MODIFY'],
            'REMOVE': cls.__constants['REMOVE'],
            'RESET': cls.__constants['RESET'],
            'ABORT': cls.__constants['ABORT'],
        }

    @property
    def EXECUTE(self):
        """Message constant 'EXECUTE'."""
        return Metaclass_MissionUpdate.__constants['EXECUTE']

    @property
    def LOAD(self):
        """Message constant 'LOAD'."""
        return Metaclass_MissionUpdate.__constants['LOAD']

    @property
    def START(self):
        """Message constant 'START'."""
        return Metaclass_MissionUpdate.__constants['START']

    @property
    def PAUSE(self):
        """Message constant 'PAUSE'."""
        return Metaclass_MissionUpdate.__constants['PAUSE']

    @property
    def RESUME(self):
        """Message constant 'RESUME'."""
        return Metaclass_MissionUpdate.__constants['RESUME']

    @property
    def STOP(self):
        """Message constant 'STOP'."""
        return Metaclass_MissionUpdate.__constants['STOP']

    @property
    def JUMP_TO(self):
        """Message constant 'JUMP_TO'."""
        return Metaclass_MissionUpdate.__constants['JUMP_TO']

    @property
    def REPEAT(self):
        """Message constant 'REPEAT'."""
        return Metaclass_MissionUpdate.__constants['REPEAT']

    @property
    def INSERT(self):
        """Message constant 'INSERT'."""
        return Metaclass_MissionUpdate.__constants['INSERT']

    @property
    def MODIFY(self):
        """Message constant 'MODIFY'."""
        return Metaclass_MissionUpdate.__constants['MODIFY']

    @property
    def REMOVE(self):
        """Message constant 'REMOVE'."""
        return Metaclass_MissionUpdate.__constants['REMOVE']

    @property
    def RESET(self):
        """Message constant 'RESET'."""
        return Metaclass_MissionUpdate.__constants['RESET']

    @property
    def ABORT(self):
        """Message constant 'ABORT'."""
        return Metaclass_MissionUpdate.__constants['ABORT']


class MissionUpdate(metaclass=Metaclass_MissionUpdate):
    """
    Message class 'MissionUpdate'.

    Constants:
      EXECUTE
      LOAD
      START
      PAUSE
      RESUME
      STOP
      JUMP_TO
      REPEAT
      INSERT
      MODIFY
      REMOVE
      RESET
      ABORT
    """

    __slots__ = [
        '_drone_id',
        '_mission_id',
        '_item_id',
        '_action',
        '_mission',
    ]

    _fields_and_field_types = {
        'drone_id': 'string',
        'mission_id': 'int32',
        'item_id': 'int32',
        'action': 'uint8',
        'mission': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.drone_id = kwargs.get('drone_id', str())
        self.mission_id = kwargs.get('mission_id', int())
        self.item_id = kwargs.get('item_id', int())
        self.action = kwargs.get('action', int())
        self.mission = kwargs.get('mission', str())

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
        if self.drone_id != other.drone_id:
            return False
        if self.mission_id != other.mission_id:
            return False
        if self.item_id != other.item_id:
            return False
        if self.action != other.action:
            return False
        if self.mission != other.mission:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def drone_id(self):
        """Message field 'drone_id'."""
        return self._drone_id

    @drone_id.setter
    def drone_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'drone_id' field must be of type 'str'"
        self._drone_id = value

    @builtins.property
    def mission_id(self):
        """Message field 'mission_id'."""
        return self._mission_id

    @mission_id.setter
    def mission_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mission_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'mission_id' field must be an integer in [-2147483648, 2147483647]"
        self._mission_id = value

    @builtins.property
    def item_id(self):
        """Message field 'item_id'."""
        return self._item_id

    @item_id.setter
    def item_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'item_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'item_id' field must be an integer in [-2147483648, 2147483647]"
        self._item_id = value

    @builtins.property
    def action(self):
        """Message field 'action'."""
        return self._action

    @action.setter
    def action(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'action' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'action' field must be an unsigned integer in [0, 255]"
        self._action = value

    @builtins.property
    def mission(self):
        """Message field 'mission'."""
        return self._mission

    @mission.setter
    def mission(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'mission' field must be of type 'str'"
        self._mission = value
