# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/TrajGenInfo.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_TrajGenInfo(type):
    """Metaclass of message 'TrajGenInfo'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'WAITING': 0,
        'EVALUATING': 1,
        'STOPPED': 2,
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
                'as2_msgs.msg.TrajGenInfo')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__traj_gen_info
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__traj_gen_info
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__traj_gen_info
            cls._TYPE_SUPPORT = module.type_support_msg__msg__traj_gen_info
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__traj_gen_info

            from as2_msgs.msg import NodeStatus
            if NodeStatus.__class__._TYPE_SUPPORT is None:
                NodeStatus.__class__.__import_type_support__()

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
            'EVALUATING': cls.__constants['EVALUATING'],
            'STOPPED': cls.__constants['STOPPED'],
        }

    @property
    def WAITING(self):
        """Message constant 'WAITING'."""
        return Metaclass_TrajGenInfo.__constants['WAITING']

    @property
    def EVALUATING(self):
        """Message constant 'EVALUATING'."""
        return Metaclass_TrajGenInfo.__constants['EVALUATING']

    @property
    def STOPPED(self):
        """Message constant 'STOPPED'."""
        return Metaclass_TrajGenInfo.__constants['STOPPED']


class TrajGenInfo(metaclass=Metaclass_TrajGenInfo):
    """
    Message class 'TrajGenInfo'.

    Constants:
      WAITING
      EVALUATING
      STOPPED
    """

    __slots__ = [
        '_header',
        '_node_status',
        '_active_status',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'node_status': 'as2_msgs/NodeStatus',
        'active_status': 'uint8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['as2_msgs', 'msg'], 'NodeStatus'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        from as2_msgs.msg import NodeStatus
        self.node_status = kwargs.get('node_status', NodeStatus())
        self.active_status = kwargs.get('active_status', int())

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
        if self.node_status != other.node_status:
            return False
        if self.active_status != other.active_status:
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
    def node_status(self):
        """Message field 'node_status'."""
        return self._node_status

    @node_status.setter
    def node_status(self, value):
        if __debug__:
            from as2_msgs.msg import NodeStatus
            assert \
                isinstance(value, NodeStatus), \
                "The 'node_status' field must be a sub message of type 'NodeStatus'"
        self._node_status = value

    @builtins.property
    def active_status(self):
        """Message field 'active_status'."""
        return self._active_status

    @active_status.setter
    def active_status(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'active_status' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'active_status' field must be an unsigned integer in [0, 255]"
        self._active_status = value
