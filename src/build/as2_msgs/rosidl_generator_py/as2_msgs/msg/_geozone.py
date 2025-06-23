# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:msg/Geozone.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Geozone(type):
    """Metaclass of message 'Geozone'."""

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
                'as2_msgs.msg.Geozone')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__geozone
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__geozone
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__geozone
            cls._TYPE_SUPPORT = module.type_support_msg__msg__geozone
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__geozone

            from geometry_msgs.msg import Polygon
            if Polygon.__class__._TYPE_SUPPORT is None:
                Polygon.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Geozone(metaclass=Metaclass_Geozone):
    """Message class 'Geozone'."""

    __slots__ = [
        '_id',
        '_alert',
        '_type',
        '_data_type',
        '_polygon',
        '_z_up',
        '_z_down',
    ]

    _fields_and_field_types = {
        'id': 'int8',
        'alert': 'int8',
        'type': 'string',
        'data_type': 'string',
        'polygon': 'geometry_msgs/Polygon',
        'z_up': 'float',
        'z_down': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Polygon'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.id = kwargs.get('id', int())
        self.alert = kwargs.get('alert', int())
        self.type = kwargs.get('type', str())
        self.data_type = kwargs.get('data_type', str())
        from geometry_msgs.msg import Polygon
        self.polygon = kwargs.get('polygon', Polygon())
        self.z_up = kwargs.get('z_up', float())
        self.z_down = kwargs.get('z_down', float())

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
        if self.id != other.id:
            return False
        if self.alert != other.alert:
            return False
        if self.type != other.type:
            return False
        if self.data_type != other.data_type:
            return False
        if self.polygon != other.polygon:
            return False
        if self.z_up != other.z_up:
            return False
        if self.z_down != other.z_down:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property  # noqa: A003
    def id(self):  # noqa: A003
        """Message field 'id'."""
        return self._id

    @id.setter  # noqa: A003
    def id(self, value):  # noqa: A003
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'id' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'id' field must be an integer in [-128, 127]"
        self._id = value

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

    @builtins.property  # noqa: A003
    def type(self):  # noqa: A003
        """Message field 'type'."""
        return self._type

    @type.setter  # noqa: A003
    def type(self, value):  # noqa: A003
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'type' field must be of type 'str'"
        self._type = value

    @builtins.property
    def data_type(self):
        """Message field 'data_type'."""
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'data_type' field must be of type 'str'"
        self._data_type = value

    @builtins.property
    def polygon(self):
        """Message field 'polygon'."""
        return self._polygon

    @polygon.setter
    def polygon(self, value):
        if __debug__:
            from geometry_msgs.msg import Polygon
            assert \
                isinstance(value, Polygon), \
                "The 'polygon' field must be a sub message of type 'Polygon'"
        self._polygon = value

    @builtins.property
    def z_up(self):
        """Message field 'z_up'."""
        return self._z_up

    @z_up.setter
    def z_up(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z_up' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'z_up' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._z_up = value

    @builtins.property
    def z_down(self):
        """Message field 'z_down'."""
        return self._z_down

    @z_down.setter
    def z_down(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z_down' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'z_down' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._z_down = value
