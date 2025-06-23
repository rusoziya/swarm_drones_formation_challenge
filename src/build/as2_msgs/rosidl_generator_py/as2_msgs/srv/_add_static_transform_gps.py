# generated from rosidl_generator_py/resource/_idl.py.em
# with input from as2_msgs:srv/AddStaticTransformGps.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_AddStaticTransformGps_Request(type):
    """Metaclass of message 'AddStaticTransformGps_Request'."""

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
                'as2_msgs.srv.AddStaticTransformGps_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__add_static_transform_gps__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__add_static_transform_gps__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__add_static_transform_gps__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__add_static_transform_gps__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__add_static_transform_gps__request

            from sensor_msgs.msg import NavSatFix
            if NavSatFix.__class__._TYPE_SUPPORT is None:
                NavSatFix.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class AddStaticTransformGps_Request(metaclass=Metaclass_AddStaticTransformGps_Request):
    """Message class 'AddStaticTransformGps_Request'."""

    __slots__ = [
        '_frame_id',
        '_child_frame_id',
        '_gps_position',
        '_azimuth',
        '_elevation',
        '_bank',
    ]

    _fields_and_field_types = {
        'frame_id': 'string',
        'child_frame_id': 'string',
        'gps_position': 'sensor_msgs/NavSatFix',
        'azimuth': 'float',
        'elevation': 'float',
        'bank': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'NavSatFix'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.frame_id = kwargs.get('frame_id', str())
        self.child_frame_id = kwargs.get('child_frame_id', str())
        from sensor_msgs.msg import NavSatFix
        self.gps_position = kwargs.get('gps_position', NavSatFix())
        self.azimuth = kwargs.get('azimuth', float())
        self.elevation = kwargs.get('elevation', float())
        self.bank = kwargs.get('bank', float())

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
        if self.frame_id != other.frame_id:
            return False
        if self.child_frame_id != other.child_frame_id:
            return False
        if self.gps_position != other.gps_position:
            return False
        if self.azimuth != other.azimuth:
            return False
        if self.elevation != other.elevation:
            return False
        if self.bank != other.bank:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def frame_id(self):
        """Message field 'frame_id'."""
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'frame_id' field must be of type 'str'"
        self._frame_id = value

    @builtins.property
    def child_frame_id(self):
        """Message field 'child_frame_id'."""
        return self._child_frame_id

    @child_frame_id.setter
    def child_frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'child_frame_id' field must be of type 'str'"
        self._child_frame_id = value

    @builtins.property
    def gps_position(self):
        """Message field 'gps_position'."""
        return self._gps_position

    @gps_position.setter
    def gps_position(self, value):
        if __debug__:
            from sensor_msgs.msg import NavSatFix
            assert \
                isinstance(value, NavSatFix), \
                "The 'gps_position' field must be a sub message of type 'NavSatFix'"
        self._gps_position = value

    @builtins.property
    def azimuth(self):
        """Message field 'azimuth'."""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'azimuth' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'azimuth' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._azimuth = value

    @builtins.property
    def elevation(self):
        """Message field 'elevation'."""
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'elevation' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'elevation' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._elevation = value

    @builtins.property
    def bank(self):
        """Message field 'bank'."""
        return self._bank

    @bank.setter
    def bank(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'bank' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'bank' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._bank = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_AddStaticTransformGps_Response(type):
    """Metaclass of message 'AddStaticTransformGps_Response'."""

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
                'as2_msgs.srv.AddStaticTransformGps_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__add_static_transform_gps__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__add_static_transform_gps__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__add_static_transform_gps__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__add_static_transform_gps__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__add_static_transform_gps__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class AddStaticTransformGps_Response(metaclass=Metaclass_AddStaticTransformGps_Response):
    """Message class 'AddStaticTransformGps_Response'."""

    __slots__ = [
        '_success',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())

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
        if self.success != other.success:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value


class Metaclass_AddStaticTransformGps(type):
    """Metaclass of service 'AddStaticTransformGps'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('as2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'as2_msgs.srv.AddStaticTransformGps')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__add_static_transform_gps

            from as2_msgs.srv import _add_static_transform_gps
            if _add_static_transform_gps.Metaclass_AddStaticTransformGps_Request._TYPE_SUPPORT is None:
                _add_static_transform_gps.Metaclass_AddStaticTransformGps_Request.__import_type_support__()
            if _add_static_transform_gps.Metaclass_AddStaticTransformGps_Response._TYPE_SUPPORT is None:
                _add_static_transform_gps.Metaclass_AddStaticTransformGps_Response.__import_type_support__()


class AddStaticTransformGps(metaclass=Metaclass_AddStaticTransformGps):
    from as2_msgs.srv._add_static_transform_gps import AddStaticTransformGps_Request as Request
    from as2_msgs.srv._add_static_transform_gps import AddStaticTransformGps_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
